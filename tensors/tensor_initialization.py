import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros((3, 3))
print(x)
x = torch.rand((3, 3))
print(x)
x = torch.ones((3, 3))
print(x)
x = torch.eye(5, 5)
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
print(x)
x = torch.diag(torch.ones(3))
print(x)

# How to initialize and convert tensors to other types (double, int, etc.)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

# Example of tensor to numpy array conversion
np_array = np.zeros((5, 5))
torch = torch.from_numpy(np_array)
print(torch)
np_array_back = torch.numpy()
print(np_array_back)