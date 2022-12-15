# IMPORTS

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters eg. relu, tanh, softmax
from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Mini batch size, convert remaining dimensions to 1D
        x = self.fc1(x)
        return x


# Checking if model is working
# model = CNN()
# x = torch.randn((64, 1, 28, 28))
# print(model(x).shape)

# SET DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# HYPER-PARAMETERS
in_channels = 1
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
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

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

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'{num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
