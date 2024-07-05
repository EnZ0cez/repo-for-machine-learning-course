import argparse
import torch
import torch.nn.functional as F  # 添加这个导入
from torch_geometric.data import DenseDataLoader
from torch.utils.data import random_split
import torch_geometric.transforms as T
from tqdm import tqdm
from math import ceil
from gnncl import Net

from utils.data_loader import FNNDataset, ToUndirected
from utils.eval_helper import eval_deep

def train_model(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.dataset == 'politifact':
        max_nodes = 500
    else:
        max_nodes = 200

        # 确保特征数量一致
    feature_num = 10

    dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset,
                         transform=T.ToDense(max_nodes), pre_transform=ToUndirected())

    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DenseDataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DenseDataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DenseDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(in_channels=feature_num, num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss_all = 0
        out_log = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _, _ = model(data.x, data.adj, data.mask)
            out_log.append([F.softmax(out, dim=1), data.y])
            loss = F.nll_loss(out, data.y.view(-1))
            loss.backward()
            loss_all += data.y.size(0) * loss.item()
            optimizer.step()
        train_metrics, loss_train = eval_deep(out_log, train_loader), loss_all / len(train_loader.dataset)

        model.eval()
        loss_val = 0
        out_log = []
        for data in val_loader:
            data = data.to(device)
            out, _, _ = model(data.x, data.adj, data.mask)
            out_log.append([F.softmax(out, dim=1), data.y])
            loss_val += data.y.size(0) * F.nll_loss(out, data.y.view(-1)).item()
        val_metrics, loss_val = eval_deep(out_log, val_loader), loss_val / len(val_loader.dataset)

        print(f'Epoch: {epoch+1}, Loss Train: {loss_train:.4f}, Loss Val: {loss_val:.4f}, '
              f'Train Metrics: {train_metrics}, Val Metrics: {val_metrics}')

    torch.save(model.state_dict(), 'gnn_model.pth')
    print('Model saved as gnn_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--epochs', type=int, default=60, help='maximum number of epochs')
    parser.add_argument('--feature', type=str, default='profile', help='feature type, [profile, spacy, bert, content]')
    args = parser.parse_args()

    train_model(args)
