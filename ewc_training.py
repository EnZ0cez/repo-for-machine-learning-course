import argparse
from tqdm import tqdm
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
from torch.utils.data import random_split

# 导入原始代码中的模型和数据加载部分
from gnncl import Net, FNNDataset, eval_deep, dataset, model, optimizer
from utils.data_loader import ToUndirected


# 定义EWC类
class EWC(object):
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        for data in self.dataloader:
            data = data.to(self.device)
            self.model.zero_grad()
            output, _, _ = self.model(data.x, data.adj, data.mask)
            loss = F.nll_loss(output, data.y.view(-1))
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n] += p.grad.detach() ** 2

        precision_matrices = {n: p / len(self.dataloader) for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

# 添加EWC版本的训练函数
def train_ewc(ewc, lambda_ewc, train_loader, device):
    model.train()
    loss_all = 0
    out_log = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _, _ = model(data.x, data.adj, data.mask)
        out_log.append([F.softmax(out, dim=1), data.y])
        loss = F.nll_loss(out, data.y.view(-1))
        loss_ewc = loss + lambda_ewc * ewc.penalty(model)
        loss_ewc.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return eval_deep(out_log, train_loader), loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(loader, device):
    model.eval()
    loss_test = 0
    out_log = []
    for data in loader:
        data = data.to(device)
        out, _, _ = model(data.x, data.adj, data.mask)
        out_log.append([F.softmax(out, dim=1), data.y])
        loss_test += data.y.size(0) * F.nll_loss(out, data.y.view(-1)).item()
    return eval_deep(out_log, loader), loss_test

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
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

lambda_values = [1, 3, 10, 30, 3e2, 3e3, 3e4, 3e5]

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

def run_experiment(train_dataset_name, test_dataset_name, output_file):
    training_set, validation_set, test_set = load_dataset(train_dataset_name, args.feature)
    train_loader = DenseDataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DenseDataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DenseDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = Net(in_channels=dataset.num_features, num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with open(output_file, 'a') as f:
        for lambda_ewc in lambda_values:
            print(f"Training with lambda: {lambda_ewc}", file=f)

            # 创建EWC对象
            ewc = EWC(model, train_loader, device)

            for epoch in tqdm(range(args.epochs)):
                train_ewc(ewc, lambda_ewc, train_loader, device)
                [acc_val, _, _, _, recall_val, auc_val, _], loss_val = test(val_loader, device)
                print(f'loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
                      f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}', file=f)

            [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = test(test_loader, device)
            print(f'Test set results with lambda={lambda_ewc}: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
                  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}', file=f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_file = 'output.txt'

# 清空之前的输出文件内容
with open(output_file, 'w') as f:
    f.write("")

# 首先在 PolitiFact 上训练，然后在 GossipCop 上训练
print("First train on PolitiFact, then on GossipCop")
run_experiment('politifact', 'gossipcop', output_file)

# 然后在 GossipCop 上训练，然后在 PolitiFact 上训练
print("First train on GossipCop, then on PolitiFact")
run_experiment('gossipcop', 'politifact', output_file)
