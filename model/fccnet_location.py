import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _flatten_input_dim(input_shape) -> int:
    # Keras used tuples; torch modules expect an integer feature size.
    return int(np.prod(input_shape))


class UserModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, labels_dim)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


class DefenseModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, labels_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DefenseOptimizeModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, labels_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        softmaxed = F.softmax(x, dim=1)
        return self.net(softmaxed)


class AttackNNModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, labels_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def model_user(input_shape, labels_dim):
    input_dim = _flatten_input_dim(input_shape)
    return UserModel(input_dim=input_dim, labels_dim=labels_dim)


def model_defense(input_shape, labels_dim):
    input_dim = _flatten_input_dim(input_shape)
    return DefenseModel(input_dim=input_dim, labels_dim=labels_dim)


def model_defense_optimize(input_shape, labels_dim):
    input_dim = _flatten_input_dim(input_shape)
    return DefenseOptimizeModel(input_dim=input_dim, labels_dim=labels_dim)


def model_attack_nn(input_shape, labels_dim):
    input_dim = _flatten_input_dim(input_shape)
    return AttackNNModel(input_dim=input_dim, labels_dim=labels_dim)