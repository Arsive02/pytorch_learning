import torch

x = torch.arange(9)
x_3x3 = x.view(3, 3)
print(x_3x3)

x_3x3 = x.reshape(3, 3)
print(x_3x3)

y = x_3x3.t()
print(y)

# print(y.view(9))

print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1,x2), dim=0))
print(torch.cat((x1,x2), dim=1))

z = x1.view(-1)
print(z)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z)
print(z.shape)

# Switch axes
z = x.permute(0, 2, 1)
print(z.shape)

x = torch.arange(10)
print(x)
print(x.unsqueeze(0))
print(x.unsqueeze(1))
print(x.unsqueeze(0).unsqueeze(1).shape)
x = x.unsqueeze(0).unsqueeze(1)
z = x.squeeze(0)
print(z.shape)
