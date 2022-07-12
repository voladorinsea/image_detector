from torch import tensor
import torch
import numpy as np

x = torch.tensor([[3]])

y = torch.tensor([[2,3,4]])
z = torch.max(x,y)
print(x)
print(y)
print(z)