import torch
import torch.nn as nn
import numpy as np
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer
from base.modules.intrinsic_motivation import IntrinsicMotivationModule
import torch.nn.functional as F
import math

class RMS(object):
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape)
        self.S = torch.ones(shape)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class APTArgs:
    def __init__(self,knn_k=16,knn_avg=True, rms=True,knn_clip=0.0005,):
        self.knn_k = knn_k 
        self.knn_avg = knn_avg 
        self.rms = rms 
        self.knn_clip = knn_clip

rms = RMS()

def compute_apt_reward(source, target, args):

    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
    reward, _ = sim_matrix.topk(args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward))  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward

class Discriminator(nn.Module, IntrinsicMotivationModule):
    def __init__(self, n, state_size, hidden_size, num_layers=4, normalize_inputs=False,
                 input_key='next_state', input_size=None, temperature=0.5, project_skill=False):
        super().__init__()
        self.temp = temperature
        self.n = n
        self.state_size = int(state_size) if input_size is None else int(input_size)
        self.input_key = str(input_key)
        
        self.state_net = nn.Sequential(nn.Linear(self.state_size, hidden_size), nn.ReLU(), 
                                        nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                                        nn.Linear(hidden_size, self.n))

        self.next_state_net = nn.Sequential(nn.Linear(self.state_size, hidden_size), nn.ReLU(), 
                                        nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                                        nn.Linear(hidden_size, self.n))

        self.pred_net = nn.Sequential(nn.Linear(2 * self.n, hidden_size), nn.ReLU(), 
                                        nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                                        nn.Linear(hidden_size, self.n))

        if project_skill:
            self.skill_net = nn.Sequential(nn.Linear(self.n, hidden_size), nn.ReLU(),
                                            nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                                            nn.Linear(hidden_size, self.n))
        else:
            self.skill_net = nn.Identity()  
   
        self.loss = self.compute_cpc_loss
        
    def forward(self, batch):
        state=batch['state']
        next_state=batch['next_state']
        skill=batch['skill']
        return self.compute_cpc_loss(state, next_state, skill).mean()
    
    def surprisal(self, batch):
        args = APTArgs()
        obs=batch['state']
        next_obs=batch['next_state']
        source = self.state_net(obs)
        target = self.state_net(next_obs)
        reward = compute_apt_reward(source, target, args) # (b,)
        return reward # (b,)
    
    # def skill_assignment(self, batch):
    #     return torch.argmax(self.layers(batch[self.input_key]), dim=1)
    
    def compute_cpc_loss(self, state, next_state, skill):
        temperature = self.temp
        eps = 1e-6
        
        assert len(state.size()) == len(next_state.size())
        skill = torch.zeros([skill.shape[0], self.n],dtype=torch.float32).scatter(-1, skill.unsqueeze(-1), 1)
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        query = self.skill_net(skill)
        query = torch.zeros_like(skill)

        key = self.pred_net(torch.cat([state,next_state],1))
        
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query,key.T) # (b,b)
        sim = torch.exp(cov / temperature) 
        neg = sim.sum(dim=-1) # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature))
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature) #(b,)
        loss = -torch.log(pos / (neg + eps)) #(b,)
        return loss
