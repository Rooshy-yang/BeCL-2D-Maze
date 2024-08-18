import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
from result_inspection.experiment import Experiment, EXPERIMENT_DIR
import matplotlib.pyplot as plt 

sys.path.append("../")

# bash
# nohup python MINE.py --algo cic_mi > log.txt 2>&1 &
# nohup python MINE.py --algo contrastive_mi > log1.txt 2>&1 &
# nohup python MINE.py --algo reverse_mi > log2.txt 2>&1 &
# nohup python MINE.py --algo forward_mi > log3.txt 2>&1 &

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--algo', type=str, default='cic_mi')         # cic_mi, contrastive_mi
parser.add_argument('--env', type=str, default='square_maze')     
args = parser.parse_args()

env = args.env
algo = args.algo

def load_exp_data(exp_name, notebook_mode=True):
    exp = Experiment(exp_name, notebook_mode=notebook_mode)
    agent = exp.learner.agent
    return exp

exp = load_exp_data(f"{args.env}/{args.algo}", notebook_mode=False)
agent = exp.learner.agent

def play_episode(agent, skill, do_eval, reset_dict={}):
    agent.reset(**reset_dict)
    agent.curr_skill = agent.curr_skill * 0 + skill
    while not agent.env.is_done:
        agent.step(do_eval)


def obtain_mine_data(NUM_TRAJECTORIES=100):
    agent = exp.learner.agent
    reset_dict = agent.env.sibling_reset

    states_list, skills_list = [], []
    for i in tqdm(range(NUM_TRAJECTORIES)):
        for skill_idx in range(agent.skill_n):
            # Collect rollout
            play_episode(agent, skill_idx, do_eval=False, reset_dict=reset_dict)
            states_list.append(np.hstack([agent.rollout[0], agent.rollout[1]]).reshape(-1))  # (102,)
            skills_list.append(np.eye(agent.skill_n)[skill_idx])  # (10,)

    return np.array(states_list), np.array(skills_list)


class Mine(nn.Module):
    def __init__(self, input_size=102+10, hidden_size=256):
        super(Mine, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc1.bias, val=0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc2.bias, val=0)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc3.bias, val=0)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        y = self.fc3(x)
        return y


def learn_mine(batch, mine_net, mine_net_optim, avg_et=1.0, unbiased_loss=True):
    batch = [torch.from_numpy(batch[0]), torch.from_numpy(batch[1])]
    joint, marginal = batch[0], batch[1]
    # calculate mutual information using MINE
    T_joint = mine_net(joint)                          # (None, 1)
    T_marginal = mine_net(marginal)                    # (None, 1)
    T_marginal_exp = torch.exp(T_marginal)
    
    # calculate loss
    if unbiased_loss:                          # unbiased gradient
        avg_et = 0.99 * avg_et + 0.01 * torch.mean(T_marginal_exp)
        # mine_estimate_unbiased = torch.mean(T_joint) - (1/avg_et).detach() * torch.mean(T_marginal_exp)
        mine_estimate = torch.mean(T_joint) - (torch.mean(T_marginal_exp)/avg_et).detach() * torch.log(torch.mean(T_marginal_exp))
        loss = -1. * mine_estimate
    else:                                      # biased gradient
        mine_estimate = torch.mean(T_joint) - torch.log(torch.mean(T_marginal_exp))
        loss = -1. * mine_estimate

    # calculate gradient and train
    mine_net.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mine_estimate, avg_et


def sample_batch(data1, data2, batch_size=100, sample_mode='joint'):
    # Construct a joint sample and a marginal sample for the two dimensions of a set of data
    # If the two dimensions in this set of data are inherently independent of each other, the distribution of joint samples and marginal samples is consistent
    assert sample_mode in ['joint', 'marginal']
    index = np.random.choice(np.arange(data1.shape[0]), size=batch_size, replace=False)
    state_batch = data1[index]           # (batch_size, 102)
    skill_batch = data2[index]           # (batch_size, 10)

    if sample_mode is 'joint':
        return np.hstack([state_batch, skill_batch]).astype(np.float32)
    elif sample_mode is 'marginal':
        # Separately sample another batch of data to extract the second dimension. Use the first dimension of the previous sample. 
        # Connect the two so that there is no dependency between the two dimensions
        new_index = np.random.choice(np.arange(data1.shape[0]), size=batch_size, replace=False)
        skill_batch_new = data2[new_index]                           
        return np.hstack([state_batch, skill_batch_new]).astype(np.float32)


def train(data1, data2, mine_net, mine_net_optim, batch_size=128, iter_num=20000, log_freq=int(1e+3)):
    result = []
    avg_et = 1.
    for i in range(iter_num):
        # Generate jointly distributed data and marginal distributed data
        joint_data = sample_batch(data1, data2, batch_size=batch_size, sample_mode='joint')
        marginal_data = sample_batch(data1, data2, batch_size=batch_size, sample_mode='marginal')
        # training the model, returning the current MI
        mi_lb, avg_et = learn_mine([joint_data, marginal_data], mine_net, mine_net_optim, avg_et)
        result.append(mi_lb.detach().numpy())
        if (i+1) % (log_freq) == 1:
            print("iter:", i, ", MI:", mi_lb.detach().numpy())
    return result


def ma(a, window_size=50):
    # Average the entire result with a sliding window
    return [np.mean(a[i: i+window_size]) for i in range(0, len(a)-window_size)]


if __name__ == '__main__':
    data1, data2 = obtain_mine_data()
    print("Data Collected:", data1.shape, data2.shape, ", algo:", algo)

    # train
    mine_net_indep = Mine()
    mine_net_optim = torch.optim.Adam(mine_net_indep.parameters(), lr=1e-3)
    result_mine = train(data1, data2, mine_net_indep, mine_net_optim)
    np.save(f"MI-result/MI-{algo}.npy", np.array(result_mine))
    
    # plot
    result_indep_ma = ma(result_indep)
    print("Mutual Information:", result_indep_ma[-1])
    plt.clf()
    plt.plot(range(len(result_indep_ma)), result_indep_ma)
    plt.title(" ".join(algo.split("_")).upper())
    plt.savefig(f"MI-result/MI-{algo}.jpg", dpi=300)

