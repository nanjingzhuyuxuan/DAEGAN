# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
class discriminator(nn.Module):
    def __init__(self, itemCount):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(itemCount, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        result = self.dis(data)
        return result


class generator(nn.Module):
    def __init__(self, train_n_item):
        super(generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(train_n_item, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),

        )
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential(
            nn.Linear(1000, train_n_item),
            nn.Tanh()
        )
    def forward(self, rating_vec):
        hidden_out = self.encoder(rating_vec)
        out = self.decoder(hidden_out)
        return out
