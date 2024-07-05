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

### GEM training

Gradient Episodic Memory (GEM): GEM uses episodic memory to store a number of samples from previous tasks, and when learning a new task t, it does not allow the loss over those samples held in memory to increase compared to when the learning of task t − 1 is finished;

**Episodic Memory Storage**:

- During the training of each new task, GEM stores a subset of samples from previous tasks in memory.
- These samples ensure that the model can refer back to previous tasks during the training of new tasks, preventing the model from forgetting previously learned knowledge.

**Computing Memory Output**:

- For the samples stored in memory, the model computes and stores their outputs.
- This step ensures that during the training of new tasks, these outputs can be accessed and compared.

**GEM Training Step**:

- In each training step, the loss for the current task is calculated.
- Simultaneously, the loss for the memory samples is computed, ensuring that this loss does not increase.
- The total loss (current task loss plus memory loss) is backpropagated to update the model parameters, ensuring that the performance on previous tasks is maintained while learning new tasks.

- use default parameters to run:

```bash
python gem_training.py
```

The program will generate a file "output_gem.txt" to store the results.

### EWC training

Elastic Weight Consolidation (EWC): its loss function consists of a quadratic penalty term on the change of the parameters, in order to prevent drastic updates to those parameters that are important to the old tasks.

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

The program will generate a file "output_ewc.txt" to store the results.

### Use own dataset

I created two files to let our datasets adjust to the model:

* The purpose of the train_model. py file is to train a Graph Neural Network (GNN) model using the given dataset and save the trained model to the file.
* The purpose of the predict_new_data.py file is to load the trained model and use it to predict new data.

Using the data from the gossipcop_v3-1_style_based_fake.json and gossipcop_v3-7_integration_based_legitimate_tn300.json datasets for  training, gossipcop_v3-5_style_based_legitimate.json dataset for prediction

1. Load new data: Load text data containing the text and label field from the gossipcop_v3-1_style_based_fake.json, gossipcop_v3-7_integration_based_legitimate_tn300.json, gossipcop_v3-5_style_based_legitimate.json file.

2. Convert to features: Use TF-IDF vectorizer to convert text data into feature vectors, ensuring consistent number of features (10).

   Use TF-IDF vectorizer to convert text data into feature vectors.

   ```python
   def text_to_features(texts):
   vectorizer = TfidfVectorizer(max_features=10)  
   X = vectorizer.fit_transform(texts)
   return torch.tensor(X.toarray(), dtype=torch.float)
   ```

3. Build adjacency matrix: Construct an identity matrix for each text as the adjacency matrix.

4. Create data objects: Encapsulate features, adjacency matrix, and mask matrix into PyTorch Geometry data objects.

5. Load trained model: Load the weights of the trained model (gnnmodel. pth).

6. Make predictions: Use models to predict new data and print the prediction results

first run the train_model.py to train and generate model.

```bash
python train_model.py --dataset politifact --feature profile --epochs 60
```

then run the predict_new_data.py to get the results.

```bash
python predict_new_data.py
```

