import json
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from gnncl import Net
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_from_file(file_path, text_field, label_field):
    texts = []
    labels = []
    label_mapping = {'legitimate': 0, 'fake': 1, 'real': 0}
    with open(file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    for item in data_json.values():
        texts.append(item[text_field])
        labels.append(label_mapping[item[label_field]])
    print(f"Loaded {len(texts)} texts from {file_path}")
    return texts, labels

def text_to_features(texts):
    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform(texts)
    print(f"Converted texts to features with shape {X.shape}")
    return torch.tensor(X.toarray(), dtype=torch.float)

def build_adj_matrix(num_nodes):
    adj = torch.eye(num_nodes)
    print(f"Built adjacency matrix with shape {adj.shape}")
    return adj

def preprocess_data(texts):
    vectorizer = TfidfVectorizer(max_features=10)
    features = vectorizer.fit_transform(texts)
    features = torch.tensor(features.toarray(), dtype=torch.float)
    data_list = []
    for feature in features:
        adj = torch.eye(1)
        mask = torch.ones(1, dtype=torch.bool)
        data_list.append(Data(x=feature.unsqueeze(0), adj=adj, mask=mask))
    return data_list

def predict(data_list, model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in data_list:
            x = data.x.unsqueeze(0) if data.x.dim() == 2 else data.x
            adj = data.adj.unsqueeze(0) if data.adj.dim() == 2 else data.adj
            mask = data.mask.unsqueeze(0) if data.mask.dim() == 1 else data.mask

            out, _, _ = model(x, adj, mask)
            pred = out.argmax(dim=1)
            predictions.extend(pred.tolist())
    return torch.tensor(predictions)

def split_data(texts, labels, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=test_size + val_size, stratify=labels)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    model_path = 'gnn_model.pth'
    model = Net(in_channels=10, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Configuration for datasets
    dataset_configs = {
        'train': [
            {
                'path': 'data/gossipcop_v3-1_style_based_fake.json',
                'text_field': 'generated_text',
                'label_field': 'generated_label'
            },
            {
                'path': 'data/gossipcop_v3-7_integration_based_legitimate_tn300.json',
                'text_field': 'doc_2_text',
                'label_field': 'doc_2_label'
            }
        ],
        'test':[
            {
                'path': 'data/gossipcop_v3-5_style_based_legitimate.json',
                'text_field': 'generated_text_t015',
                'label_field': 'generated_label'
            }
        ]
    }

    # Load and preprocess data
    all_texts, all_labels = [], []
    for config in dataset_configs['train']:
        texts, labels = load_data_from_file(config['path'], config['text_field'], config['label_field'])
        all_texts.extend(texts)
        all_labels.extend(labels)

    # Split data into training, validation, and test sets
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(all_texts, all_labels)

    # Preprocess data
    train_data = preprocess_data(train_texts)
    val_data = preprocess_data(val_texts)
    test_data = preprocess_data(test_texts)

    print(f"Training label distribution: {Counter(train_labels)}")
    print(f"Validation label distribution: {Counter(val_labels)}")
    print(f"Test label distribution: {Counter(test_labels)}")

    # Make predictions on the validation set
    val_predictions = predict(val_data, model)
    val_accuracy = accuracy_score(val_labels, val_predictions.numpy())
    val_precision = precision_score(val_labels, val_predictions.numpy(), zero_division=1)
    val_recall = recall_score(val_labels, val_predictions.numpy(), zero_division=1)
    val_f1 = f1_score(val_labels, val_predictions.numpy(), zero_division=1)
    print(f"Validation - Accuracy: {val_accuracy}")


    # Make predictions on the test set
    test_predictions = predict(test_data, model)
    test_accuracy = accuracy_score(test_labels, test_predictions.numpy())
    test_precision = precision_score(test_labels, test_predictions.numpy(), zero_division=1)
    test_recall = recall_score(test_labels, test_predictions.numpy(), zero_division=1)
    test_f1 = f1_score(test_labels, test_predictions.numpy(), zero_division=1)
    print(f"Test - Accuracy: {test_accuracy}")

