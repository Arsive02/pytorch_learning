import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([5, 6, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
print(z2)

z = x + y
print(z)

# Subtraction
z = x - y
print(z)

# Division
z = torch.true_divide(x, y)  # Element wise
print(z)

# inplace operations
t = torch.zeros(3)
t.add_(x)  # Computationally efficient
print(t)

t += x  # Not computationally efficient

# Exponentiation
z = x.pow(2)
print(z)

z = x ** 2
print(z)

# Simple comparison
z = x > 0
print(z)

z = x < 0
print(z)

# Matrix multiplication
x = torch.rand((2, 3))
y = torch.rand((3, 2))
z = torch.mm(x, y)
print(z)

x3 = x.mm(y)
print(x3)

# Matrix Exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# Element wise multiplication
x = torch.rand((2, 2))
y = torch.rand((2, 2))
z = x * y
print(z)

# Dot product
x = torch.tensor([1, 2, 3])
y = torch.tensor([5, 6, 7])
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 30
p = 20
x1 = torch.rand((batch, n, m))
x2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(x1, x2)  # (batch, n, p)
print(out_bmm.shape)

# Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
print(z)

z = x1 ** x2
print(z)

# Useful Tensor operations
sum_x = torch.sum(x1, dim=0)
print(sum_x)
