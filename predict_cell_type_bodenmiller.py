# import pandas as pd
import polars as pl

cell_types = pl.read_csv("/juno/work/shah/users/ibrahih3/codebase/space-gm/data_bodenmiller/basel_data/BaselTMA_SP41_2_X2Y8_cell_types.csv")
expressions = pl.read_csv("/juno/work/shah/users/ibrahih3/codebase/space-gm/data_bodenmiller/basel_data/BaselTMA_SP41_2_X2Y8_expression.csv")

# # join cell types and expression on CELL_ID

data = cell_types.join(expressions, on="CELL_ID")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# define the network

class CellClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CellClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
    
# define the data loader
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# assuming 'data' is your DataFrame

# select only the biomarker columns
biomarkers = [col for col in data.columns if col not in ['CELL_ID', 'CLUSTER_LABEL', 'ACQUISITION_ID', 'CELL_TYPE']]
X = data[biomarkers].to_pandas().values

# convert the biomarker data into PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float)
    
    
# convert the cell type labels into integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['CELL_TYPE'].to_pandas())  # convert to pandas for label encoding

# convert the labels into PyTorch tensors
y_tensor = torch.tensor(y, dtype=torch.long)

# Combine the tensors
dataset = TensorDataset(X_tensor, y_tensor)

# Split the data into training and test sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define your dataloaders
batch_size = 32  # choose the batch size that suits your needs
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    
# Define the model

input_size = 30 # data.shape[1] - 3
hidden_size = 50  # could be optimized
num_classes = len(data["CELL_TYPE"].unique())
    
model = CellClassifier(input_size, hidden_size, num_classes)

# loss_function = nn.NLLLoss()

# Define the optimizer - Stochastic Gradient Descent
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
# for epoch in range(100):  # number of epochs
#     for inputs, targets in train_dataloader:  
#         # Clear the gradients from the last step
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs)

#         # Compute the loss
#         loss = loss_function(outputs, targets)

#         # Backward pass and update
#         loss.backward()
#         optimizer.step()

#     # Print the loss for this epoch
#     print('Epoch', epoch, 'Loss', loss.item())

def train_model(model, train_dataloader, num_epochs=100, print_loss=True):
    """
    Train the model
    
    Parameters:
        model (nn.Module): the model to be trained
        train_dataloader (DataLoader): the dataloader for the training data
    """
    
    # Define the loss function
    loss_function = nn.NLLLoss()

    # Define the optimizer - Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):  # number of epochs
        for inputs, targets in train_dataloader:
            # inputs, targets = inputs.to(device), targets.to(device)        

            # Clear the gradients from the last step
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = loss_function(outputs, targets)

            # Backward pass and update
            loss.backward()
            optimizer.step()

        if print_loss:
            # Print the loss for this epoch
            if epoch % 10 == 0:  # Only print every 10 epochs
                print('Epoch', epoch, 'Loss', loss.item())

def test_model(model, test_dataloader, print_acc=True, return_acc=True, label_encoder=None):
    """
    Evaluate the model
    
    Parameters:
        model (nn.Module): the model to be evaluated
        test_dataloader (DataLoader): the dataloader for the test data
    """
    
    # switch the model to evaluation mode
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # No need to track gradients for validation, saves memory and computations
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    if print_acc: print('Test Accuracy: %.2f %%' % (100 * accuracy))
    
    if label_encoder:
        all_preds = label_encoder.inverse_transform(all_preds)
        all_labels = label_encoder.inverse_transform(all_labels)
        
    if return_acc: return all_labels, all_preds, accuracy
    
    else: return all_labels, all_preds



train_model(model, train_dataloader, num_epochs=200)
_ = test_model(model, test_dataloader)

# model.train()
# for epoch in range(100):
#     for batch_idx, (data, target) in enumerate(train_dataloader):
#         # data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = loss_function(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_dataloader.dataset),
#                 100. * batch_idx / len(train_dataloader), loss.item()))
    
    
# Evaluate the model

# switch the model to evaluation mode
# model.eval()

# correct = 0
# total = 0

# # No need to track gradients for validation, saves memory and computations
# with torch.no_grad():
#     for inputs, labels in test_dataloader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Test Accuracy: %d %%' % (100 * correct / total))


# for the whole dataset (Melanoma_Final)

# data_melanoma_final = pl.read_csv("/juno/work/shah/users/ibrahih3/melanoma/Melanoma_FINAL.csv")

# # use the needed columns only -- CELL_ID, CLUSTER_LABEL, ACQUISITION_ID, and biomarkers
# # biomarkers = [col for col in data_melanoma_final.columns if col in ['CELL_ID', 'CLUSTER_LABEL', 'ACQUISITION_ID']]
# biomarkers = ['CD45', 'CD3', 'CD4', 'CD8', 'FOXP3', 'CD56', 'CD20', 'CD14', 'CD163', 'CD68', 
#               'SOX10', 'S100B', 'KI67', 'CD25', 'PD1', 'LAG3', 'TIM3', 'CD27', 'PDL1', 'B7H3', 
#               'IDO1', 'B2M', 'MHCI', 'MHCII', 'MRC1', 'TGM2']
# X = data_melanoma_final[biomarkers].to_pandas().values

# # convert the biomarker data into PyTorch tensors
# X_tensor = torch.tensor(X, dtype=torch.float)
    
# # convert the cell type labels into integers
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(data_melanoma_final['celltype'].to_pandas())  # convert to pandas for label encoding

# # convert the labels into PyTorch tensors
# y_tensor = torch.tensor(y, dtype=torch.long)

# # Combine the tensors
# dataset = TensorDataset(X_tensor, y_tensor)

# # Split the data into training and test sets
# train_size = int(0.8 * len(dataset))  # 80% for training
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# # Define your dataloaders
# batch_size = 32  # choose the batch size that suits your needs
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# # Define the model

# input_size = 26
# hidden_size = 50  # could be optimized
# num_classes = len(data_melanoma_final["celltype"].unique())
    
# model = CellClassifier(input_size, hidden_size, num_classes)

# loss_function = nn.NLLLoss()

# # Define the optimizer - Stochastic Gradient Descent
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # Training loop
# for epoch in range(100):  # number of epochs
#     for inputs, targets in train_dataloader:  
#         # Clear the gradients from the last step
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs)

#         # Compute the loss
#         loss = loss_function(outputs, targets)

#         # Backward pass and update
#         loss.backward()
#         optimizer.step()

#     # Print the loss for this epoch
#     print('Epoch', epoch, 'Loss', loss.item())
    
    
# # Evaluate the model

# # switch the model to evaluation mode
# model.eval()

# correct = 0
# total = 0

# # No need to track gradients for validation, saves memory and computations
# with torch.no_grad():
#     for inputs, labels in test_dataloader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Test Accuracy: %d %%' % (100 * correct / total))



import pickle

import networkx as nx
import dgl

file = "/juno/work/shah/users/pourmalm/mpif_data/melanoma/gnn_output/v2/3_train_graphs_targets_cohort.pkl"

train_graphs_targets = pickle.load(open(file, "rb"))
print(type(train_graphs_targets)) # This should return <class 'tuple'>

first_list = train_graphs_targets[0] 
print(type(first_list)) # This should return <class 'list'>

first_graph = first_list[0]
print(type(first_graph)) # This should return <class 'dgl.heterograph.DGLHeteroGraph'>

import networkx as nx
import matplotlib.pyplot as plt

# Convert DGLHeteroGraph to networkx graph for visualization
nxg = first_graph.to_networkx()

# Draw the graph
plt.figure(figsize=(10,10))
nx.draw(nxg, with_labels=True)
plt.show()
