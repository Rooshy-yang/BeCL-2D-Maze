import torch
import torch.nn as nn
import numpy as np
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer
from base.modules.intrinsic_motivation import IntrinsicMotivationModule
import torch.nn.functional as F
import math

class Discriminator(nn.Module, IntrinsicMotivationModule):
    def __init__(self, n, state_size, hidden_size, num_layers=4, normalize_inputs=False,
                 input_key='next_state', input_size=None, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.n = n
        self.state_size = int(state_size) if input_size is None else int(input_size)
        self.input_key = str(input_key)
        assert num_layers >= 2
        self.num_layers = int(num_layers)

        input_normalizer = Normalizer(self.state_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=self.state_size, output_size=self.n, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer)

        self.loss = self.compute_info_nce_loss
        
    def forward(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)
        ns = batch['next_state']
        for layer in self.layers:
            ns = layer(ns)
        return self.loss(x, ns).mean()
    
    def surprisal(self, batch):
        s = batch[self.input_key]
        for layer in self.layers:
            s = layer(s)
        ns = batch['next_state']
        for layer in self.layers:
            ns = layer(ns)
        return torch.exp(-self.compute_info_nce_loss(s, ns)).squeeze()
    
    # def skill_assignment(self, batch):
    #     return torch.argmax(self.layers(batch[self.input_key]), dim=1)
    
    def compute_info_nce_loss(self, features, ns_features):
        # label positives samples
        temperature = self.temperature
        eps = 1e-6
        features = F.normalize(features, dim=1)
        ns_features = F.normalize(ns_features, dim=1)
        cov = torch.mm(features,ns_features.T) # (b,b)
        sim = torch.exp(cov / temperature) 
        neg = sim.sum(dim=-1) # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature))
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(features * ns_features, dim=-1) / temperature) #(b,)
        loss = -torch.log(pos / (neg + eps)) #(b,)
        
        return loss
