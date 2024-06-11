# Reproduced-Continual-Learning-for-Fake-News-Detection-from-Social-Media

This repo includes the Pytorch-Geometric implementation of a Graph Neural Network (GNN-CL) based fake news detection model.

Paper:Han, Yi, Shanika Karunasekera, and Christopher Leckie. "Graph neural networks with continual learning for fake news detection from social media." arXiv preprint arXiv:2007.03316 (2020).https://arxiv.org/pdf/2007.03316.pdf

## Installation

To run the code in this repo, you need to have `Python>=3.6`, `PyTorch>=1.6`, and `PyTorch-Geometric>=1.6.1`. Please follow the installation instructions of [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) to install PyG.

## Run

Here I provide three ways to run the model.

### Use single dataset

```python
num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

```

* 20% of the dataset are used to train
* 10% of the dataset are used to validate
* 70% of the dataset are used to test

use default parameters to run:

```bash
python gnncl.py
```

use custom parameters to run:

```bash
python gnncl.py --seed 777 --dataset gossipcop --batch_size 64 --lr 0.01 --weight_decay 0.001 --nhid 128 --epochs 60 --feature content
```

### EWC training

- **Training the Initial Model on the PolitiFact Dataset:**

  - The initial model is trained on the PolitiFact dataset to obtain the optimal parameters $ \theta_{PolitiFact}^{*} $ and the corresponding Fisher information matrix $ F_{PolitiFact} $.

  **Incremental Training:**

  - Incremental training is then performed on the GossipCop dataset. The loss function includes the loss on the GossipCop dataset and a regularization term based on the Fisher information matrix and optimal parameters from the PolitiFact dataset.

  **Hyperparameter Adjustment:**

  - During training, the regularization coefficient $ \lambda $ and sample size $ |M| $ are adjusted to balance the performance on both datasets.

  use default parameters to run:

```bash
python ewc_training.py
```

The program will generate a file "output.txt" to store the results.

### Use own dataset

I created two files to let our datasets adjust to the model:

* The purpose of the train_model. py file is to train a Graph Neural Network (GNN) model using the given dataset and save the trained model to the file.
* The purpose of the predict_new_data.py file is to load the trained model and use it to predict new data.

Use TF-IDF vectorizer to convert text data into feature vectors.

```python
def text_to_features(texts):
vectorizer = TfidfVectorizer(max_features=10)  
X = vectorizer.fit_transform(texts)
return torch.tensor(X.toarray(), dtype=torch.float)
```

first run the train_model.py to get the model file.

```bash
python train_model.py --dataset politifact --feature profile --epochs 60
```

then run the predict_new_data.py to get the results.

```bash
python predict_new_data.py
```

