import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# SET DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# HYPER-PARAMETERS
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2


# CREATE RNN CLASS
class BiRNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Dimensions -> (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward prop
        out, _ = self.lstm(x, (h0, c0))         # _ is cell state and hidden state
        out = out[:, -1, :]
        out = self.fc(out)

        return out

train_dataset = datasets.MNIST(root="dataset/",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_dataset = datasets.MNIST(root="dataset/",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

model = BiRNN(input_size, num_layers, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TRAINING LOOP
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # FORWARD
        scores = model(data)
        loss = criterion(scores, targets)

        # OPTIMIZE
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


# CHECK ACCURACY
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
