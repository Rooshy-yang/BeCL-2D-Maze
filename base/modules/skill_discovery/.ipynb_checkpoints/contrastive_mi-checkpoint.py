import torch
import torch.nn as nn
import numpy as np
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer
from base.modules.intrinsic_motivation import IntrinsicMotivationModule
import torch.nn.functional as F

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
                 input_key='next_state', input_size=None, temperature=0.01):
        super().__init__()
        self.temperature = temperature
        self.n = n
        self.state_size = int(state_size) if input_size is None else int(input_size)
        self.input_key = str(input_key)
        assert num_layers >= 2
        self.num_layers = int(num_layers)
        self.count = 0 
        input_normalizer = Normalizer(self.state_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=self.state_size, output_size=self.n, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer)

        self.loss = self.compute_info_nce_loss
        
    def forward(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)
        return self.loss(x, batch['skill']).mean()
    
    def surprisal(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)

        args = APTArgs()
        self.count += 1
        return compute_apt_reward(x, x, args)
#         else: 
#             return torch.exp(-self.compute_info_nce_loss(x, batch['skill'])).squeeze()
    
    # def skill_assignment(self, batch):
    #     return torch.argmax(self.layers(batch[self.input_key]), dim=1)
    
    def compute_info_nce_loss(self, features, labels):
        # label positives samples
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long()

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix -= torch.max(similarity_matrix, 1)[0][:, None]
        similarity_matrix = torch.exp(similarity_matrix)

        # update not only when all negative sample existed
        # assert labels[pick_one_positive_sample_idx]
        pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
        pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)

        # select one and combine multiple positives
        positives = torch.sum(similarity_matrix * pick_one_positive_sample_idx, dim=-1, keepdim=True)
        negatives = torch.sum(similarity_matrix, dim=-1, keepdim=True)
        eps = torch.as_tensor(1e-6)
        loss = -torch.log(positives / (negatives + eps) + eps)

        return loss
