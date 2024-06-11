# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def cnn_7layer_bn2(input_ch=3, input_dim=32, width=64, linear_size=512, output_size=10):
    model = nn.Sequential(
        nn.Conv2d(input_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear((input_dim // 2) * (input_dim // 2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, output_size),
    )
    return model


def cnn(input_ch=3, input_dim=32):
    return cnn_7layer_bn2(input_ch, input_dim)


def cnn_7layer_bn_imagenet(input_ch=3, input_dim=64, width=64, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(input_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32768, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 200),
    )
    return model


def cnn_7layer_bn_wolast(
    input_ch=3, input_dim=32, width=64, linear_size=512, output_size=10
):
    model = nn.Sequential(
        nn.Conv2d(input_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear((input_dim // 2) * (input_dim // 2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, output_size),
    )
    return model
