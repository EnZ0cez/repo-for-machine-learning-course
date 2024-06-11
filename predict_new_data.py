import json
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from gnncl import Net

def load_new_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    texts = []
    for item in data_json.values():
        texts.append(item['generated_text_glm4'])  # 使用 generated_text_glm4 字段

    return texts

def text_to_features(texts):
    vectorizer = TfidfVectorizer(max_features=10)  # 确保特征数一致
    X = vectorizer.fit_transform(texts)
    return torch.tensor(X.toarray(), dtype=torch.float)

def build_adj_matrix(num_nodes):
    adj = torch.eye(num_nodes)
    return adj

def preprocess_new_data(input_file):
    texts = load_new_data(input_file)
    features = text_to_features(texts)
    adj = build_adj_matrix(features.size(0))
    mask = torch.ones(features.size(0), dtype=torch.bool)

    return Data(x=features, adj=adj, mask=mask)

def predict_new_data(input_file):
    data = preprocess_new_data(input_file)
    model = Net(in_channels=10, num_classes=2)  # 确保输入特征数一致
    model.load_state_dict(torch.load('gnn_model.pth'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        out, _, _ = model(data.x.unsqueeze(0), data.adj.unsqueeze(0), data.mask.unsqueeze(0))
        pred = F.softmax(out, dim=1).argmax(dim=1)
        print(f'Prediction: {pred.item()}')

if __name__ == '__main__':
    input_file = 'D:\\SZU\\机器学习\\gossipcop\\GNN-FakeNews-main\\data\\gossipcop_v3-2_content_based_fake.json'
    predict_new_data(input_file)
