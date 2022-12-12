import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([5, 6, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)
