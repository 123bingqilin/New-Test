import torch
import torch.nn as nn
from .util import mlp


class ForwardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *([hidden_dim] * n_hidden), obs_dim])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)


class InverseModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim * 2, *([hidden_dim] * n_hidden), act_dim])

    def forward(self, obs, next_obs):
        x = torch.cat([obs, next_obs], dim=1)
        return self.net(x)


class SharedEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), hidden_dim])

    def forward(self, obs):
        return self.net(obs)


class SharedForwardModel(nn.Module):
    def __init__(self, feat_dim, act_dim, obs_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([feat_dim + act_dim, *([hidden_dim] * n_hidden), obs_dim])

    def forward(self, feat, act):
        x = torch.cat([feat, act], dim=1)
        return self.net(x)


class SharedInverseModel(nn.Module):
    def __init__(self, feat_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([feat_dim * 2, *([hidden_dim] * n_hidden), act_dim])

    def forward(self, feat, next_feat):
        x = torch.cat([feat, next_feat], dim=1)
        return self.net(x)