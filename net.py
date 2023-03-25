import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, conv1_out, conv1_kernel_size, pooling_kernel_size, conv2_out, conv2_kernel_size, fc2_in, fc3_in):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_out, conv1_kernel_size)
        self.pool = nn.MaxPool2d(pooling_kernel_size)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, conv2_kernel_size)

        init_shape = (32, 32)
        conv1_out_shape = (init_shape[0] - (conv1_kernel_size - 1), init_shape[1] - (conv1_kernel_size - 1))
        H, W = conv1_out_shape
        k = pooling_kernel_size
        conv2_in_shape = np.floor((H-k)/k) + 1, np.floor((W-k)/k) + 1

        conv2_out_shape = (conv2_in_shape[0] - (conv2_kernel_size - 1), conv2_in_shape[1] - (conv2_kernel_size - 1))
        H, W = conv2_out_shape
        fc1_in_shape = np.floor((H-k)/k) + 1, np.floor((W-k)/k) + 1


        self.fc1 = nn.Linear(conv2_out * fc1_in_shape[0] * fc1_in_shape[1], fc2_in)
        self.fc2 = nn.Linear(fc2_in, fc3_in)
        self.fc3 = nn.Linear(fc3_in, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x