import argparse
from tqdm import tqdm
from math import ceil
import random
import logging

import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
from torch.utils.data import random_split

from gnncl import Net, GNN, dense_diff_pool
from utils.data_loader import FNNDataset, ToUndirected
from utils.eval_helper import eval_deep

# Set up logging
logging.basicConfig(filename='output_gem.txt', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


class GEM(object):
    def __init__(self, model, dataset, sample_size, device):
        self.model = model
        self.device = device
        data_list = [dataset[i] for i in range(len(dataset))]
        sample_size = min(sample_size, len(data_list))  # Ensure sample size does not exceed dataset size
        self.memory_data = random.sample(data_list, sample_size)
        self.memory_output = self.compute_memory_output()

    def compute_memory_output(self):
        memory_output = []
        self.model.eval()
        with torch.no_grad():
            for data in self.memory_data:
                data = data.to(self.device)
                x = data.x.unsqueeze(0) if len(data.x.size()) == 2 else data.x
                adj = data.adj.unsqueeze(0) if len(data.adj.size()) == 2 else data.adj
                mask = data.mask.unsqueeze(0) if len(data.mask.size()) == 1 else data.mask
                output, _, _ = self.model(x, adj, mask)
                memory_output.append(output)
        return memory_output

    def gem_step(self, loss):
        memory_loss = 0
        self.model.eval()
        for output in self.memory_output:
            memory_loss += F.mse_loss(output, output.detach())
        memory_loss /= len(self.memory_output)

        total_loss = loss + memory_loss
        total_loss.backward()


def train_gem(model, train_loader, gem, optimizer, device):
    model.train()
    loss_all = 0
    out_log = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x.unsqueeze(0) if len(data.x.size()) == 2 else data.x
        adj = data.adj.unsqueeze(0) if len(data.adj.size()) == 2 else data.adj
        mask = data.mask.unsqueeze(0) if len(data.mask.size()) == 1 else data.mask
        out, _, _ = model(x, adj, mask)
        out_log.append([F.softmax(out, dim=1), data.y])


        loss = F.nll_loss(out, data.y.view(-1))
        gem.gem_step(loss)
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return eval_deep(out_log, train_loader), loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader, model, device):
    model.eval()
    loss_test = 0
    out_log = []
    for data in loader:
        data = data.to(device)
        x = data.x.unsqueeze(0) if len(data.x.size()) == 2 else data.x
        adj = data.adj.unsqueeze(0) if len(data.adj.size()) == 2 else data.adj
        mask = data.mask.unsqueeze(0) if len(data.mask.size()) == 1 else data.mask
        out, _, _ = model(x, adj, mask)
        out_log.append([F.softmax(out, dim=1), data.y])
        loss_test += data.y.size(0) * F.nll_loss(out, data.y.view(-1)).item()
    return eval_deep(out_log, loader), loss_test


def load_dataset(name, feature):
    if name == 'politifact':
        max_nodes = 500
    else:
        max_nodes = 200
    dataset = FNNDataset(root='data', feature=feature, empty=False, name=name,
                         transform=T.ToDense(max_nodes), pre_transform=ToUndirected())
    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    return training_set, validation_set, test_set


def run_experiment(train_dataset_name, test_dataset_name, sample_sizes, output_file):
    for sample_size in sample_sizes:
        training_set, validation_set, test_set = load_dataset(train_dataset_name, args.feature)
        train_loader = DenseDataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DenseDataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DenseDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = Net(in_channels=training_set[0].x.size(1), num_classes=len(torch.unique(training_set.dataset.data.y))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        gem = GEM(model, training_set, sample_size, device)

        for epoch in tqdm(range(args.epochs)):
            [acc_train, _, _, _, recall_train, auc_train, _], loss_train = train_gem(model, train_loader, gem, optimizer, device)
            [acc_val, _, _, _, recall_val, auc_val, _], loss_val = test(val_loader, model, device)
            logger.info(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
                        f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
                        f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
                        f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

        [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = test(test_loader, model, device)
        logger.info(f'Test set results with sample_size={sample_size}: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
                    f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')


# Initialize parameters
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=60, help='maximum number of epochs')
parser.add_argument('--feature', type=str, default='profile', help='feature type, [profile, spacy, bert, content]')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_file = 'output_gem.txt'

# Clear previous output file content
with open(output_file, 'w') as f:
    f.write("")

# Define sample sizes
sample_sizes = [100, 200, 300]

# First train on PolitiFact, then on GossipCop
print("First train on PolitiFact, then on GossipCop")
logger.info("First train on PolitiFact, then on GossipCop")
run_experiment('politifact', 'gossipcop', sample_sizes, output_file)

# Then train on GossipCop, then on PolitiFact
print("First train on GossipCop, then on PolitiFact")
logger.info("First train on GossipCop, then on PolitiFact")
run_experiment('gossipcop', 'politifact', sample_sizes, output_file)
