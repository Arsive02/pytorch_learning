# IMPORTS

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters eg. relu, tanh, softmax
from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

from tqdm import tqdm


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# Checking if model is working
# model = NN(28 * 28, 10)
# x = torch.rand((64, 784))  # (mini-batch, input_size)
# print(model(x).shape)      # (mini-batch, num_classes)

# SET DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# HYPER-PARAMETERS
input_size = 784
num_classes = 10
num_batches = 64
epochs = 10
learning_rate = 0.001

# LOAD DATA
train_dataset = datasets.MNIST(root="dataset/",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=num_batches,
                          shuffle=True)

test_dataset = datasets.MNIST(root="dataset/",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=num_batches,
                         shuffle=True)

# INITIALIZE NETWORK
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# LOSSES AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TRAIN NETWORK
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Set the data and target to the device
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Set the correct shape
        data = data.reshape([data.shape[0], -1])

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()


# CHECK ACCURACY ON TRAINING & TEST TO SEE HOW GOOD OUR MODEL
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'{num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

