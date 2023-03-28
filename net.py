import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, conv1_out, conv1_kernel_size, pooling_kernel_size, conv2_out, conv2_kernel_size, fc2_in, fc3_in):
        super().__init__()

        # print(f'conv1: 3, {conv1_out}, {conv1_kernel_size}')
        # print(f'pool: {pooling_kernel_size}')
        # print(f'conv2: {conv1_out}, {conv2_out}, {conv2_kernel_size}')

        self.conv1 = nn.Conv2d(3, conv1_out, conv1_kernel_size)
        self.pool = nn.MaxPool2d(pooling_kernel_size)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, conv2_kernel_size)

        init_shape = (32, 32)
        conv1_out_shape = (init_shape[0] - (conv1_kernel_size - 1), init_shape[1] - (conv1_kernel_size - 1))
        # print(f'conv1_out_shape: {conv1_out_shape}')
        conv2_in_shape = (conv1_out_shape[0] // pooling_kernel_size, conv1_out_shape[1] // pooling_kernel_size)
        # print(f'conv2_in_shape: {conv2_in_shape}')

        conv2_out_shape = (conv2_in_shape[0] - (conv2_kernel_size - 1), conv2_in_shape[1] - (conv2_kernel_size - 1))
        # print(f'conv2_out_shape: {conv2_out_shape}')
        
        fc1_in_shape = (conv2_out_shape[0] // pooling_kernel_size, conv2_out_shape[1] // pooling_kernel_size)
        # print(f'fc1_in_shape: {fc1_in_shape}')

        # print(f'fc1: {conv2_out} * {fc1_in_shape[0]} * {fc1_in_shape[1]}, {fc2_in}')
        # print(f'fc2: {fc2_in}, {fc3_in}')
        # print(f'fc3: {fc3_in}, 10')

        self.fc1 = nn.Linear(conv2_out * fc1_in_shape[0] * fc1_in_shape[1], fc2_in)
        self.fc2 = nn.Linear(fc2_in, fc3_in)
        self.fc3 = nn.Linear(fc3_in, 10)

    def forward(self, x):
        # print('initial x shape:', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print('x shape after conv1+relu+pool:', x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print('x shape after conv2+relu+pool:', x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print('x shape after flatten:', x.shape)
        x = F.relu(self.fc1(x))
        # print('x shape after fc1:', x.shape)
        x = F.relu(self.fc2(x))
        # print('x shape after fc2:', x.shape)
        x = self.fc3(x)
        # print('final x shape:', x.shape)
        return x