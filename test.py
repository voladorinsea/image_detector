from torch import tensor
import torch
import numpy as np
grid_size = 7
grid = np.arange(grid_size)
a,b = np.meshgrid(grid, grid)

x_offset = torch.FloatTensor(a).view(-1,1)
y_offset = torch.FloatTensor(b).view(-1,1)

x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,3).view(-1,2)

print(x_y_offset)