# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import sys
import math
import json
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from result_inspection.toy_maze import load_smm_buffer
from agents.maze_agents.toy_maze.env.maze_env import Env
from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator


def sample_dataset(maze_type, num_samples, condition_fn=lambda x: True):
    """ condition_fn can be used to induce a prior over samples """
    env = Env(n=50, maze_type=maze_type, use_antigoal=False)
    dataset = np.zeros((num_samples, 2))
    for sample_idx in range(num_samples):
        done = False
        while not done:
            s = env.sample()
            done = condition_fn(s)
        dataset[sample_idx] = np.array(s)
    dataset = torch.from_numpy(dataset).float()
    return env, dataset


def open_experiment():
    parser = argparse.ArgumentParser("Train VQ-VAE.")
    parser.add_argument('--config-path', type=str, help='Path to experiment config file (expecting a json)')
    parser.add_argument('--log-dir', type=str, help='Parent directory that holds experiment log directories')
    parser.add_argument('--dur', type=int, default=50000, help='Number of training iterations')
    args = parser.parse_args()

    config_path = args.config_path
    assert os.path.isfile(config_path)
    config = json.load(open(config_path))

    exp_name = config_path.split('/')[-1][:-5]
    exp_dir = os.path.join(args.log_dir, exp_name)

    print('Experiment directory is: {}'.format(exp_dir), flush=True)

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
        shutil.copyfile(config_path, os.path.join(exp_dir, 'config.json'))

    if config['sampler'] == 'oracle':
        env, dataset = sample_dataset(config['maze_type'], config['num_samples'])
    elif config['sampler'] == 'smm':
        exp, dataset = load_smm_buffer(config['smm_exp_name'], config['smm_epoch'], notebook_mode=False)
        env = exp.learner.agent.env
        config['maze_type'] = env.maze_type
    else:
        raise ValueError("Invalid 'sampler' type")

    # Create VQ-VAE model and compute moments for the normalizer module
    model = VQVAEDiscriminator(state_size=env.state_size, **config['vae_args'])
    model.update_normalizer(dataset=dataset)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return model, optimizer, dataset, config, args, exp_dir


if __name__ == '__main__':
    # Interpret the arguments. Create the model, optimizer and dataset. Fetch the config file.
    model, optim, dataset, config, args, save_dir = open_experiment()

    # Training loop
    indices = list(range(dataset.size(0)))
    loss_list = []
    model.train()
    for iter_idx in tqdm(range(args.dur), desc="Training"):
        # Make batch
        batch_indices = np.random.choice(indices, size=config['batch_size'])
        batch = dict(next_state=dataset[batch_indices])

        # Forward + backward pass
        optim.zero_grad()
        loss = model(batch)
        loss.backward()
        optim.step()

        # Log progress
        loss_list.append(loss.item())

    # Save model, config and losses
    model.eval()
    model_path = os.path.join(save_dir, "model.pth.tar")
    config_path = os.path.join(save_dir, "config.json")
    loss_path = os.path.join(save_dir, "loss.json")
    torch.save(model.state_dict(), model_path)
    with open(config_path, 'wt') as f:
        json.dump(config, f)
    with open(loss_path, 'wt') as f:
        json.dump(json.dumps(loss_list), f)
