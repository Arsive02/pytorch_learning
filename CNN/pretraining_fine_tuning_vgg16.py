# IMPORTS
import sys
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters eg. relu, tanh, softmax
import torchvision.models
from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

from tqdm import tqdm

# SET DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# HYPER-PARAMETERS
in_channels = 3
num_classes = 10
num_batches = 1024
epochs = 5
learning_rate = 0.001


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# INITIALIZE NETWORK
model = torchvision.models.vgg16(pretrained=True)
model.avgpool = Identity()
model.classifier = nn.Linear(512, num_classes)
print(model)
model.to(device)

# LOSSES AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# LOAD DATA
train_dataset = datasets.CIFAR10(root="dataset/",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=num_batches,
                          shuffle=True)

test_dataset = datasets.CIFAR10(root="dataset/",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=num_batches,
                         shuffle=True)



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
