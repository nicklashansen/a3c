import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd import Variable
import gym
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import pickle
import time
from collections import deque

'''
@authors:
Nicklas Hansen,
Peter Ebert Christensen
'''

# Load CUDA
CUDA = False#torch.cuda.is_available()
print('CUDA has been enabled.' if CUDA is True else 'CUDA has been disabled.')

# Define tensors
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
IntTensor   = torch.cuda.IntTensor if CUDA else torch.IntTensor
LongTensor  = torch.cuda.LongTensor if CUDA else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor if CUDA else torch.ByteTensor
Tensor = FloatTensor

# Set global datetime
dt = str(datetime.datetime.now()).split('.')[0].replace(' ', '-')[5:]


class Agent(nn.Module):
    # Interface for a neural RL agent
    def __init__(self, args):
        super(Agent, self).__init__()
        _env = gym.make(args.env)
        self.size_in = _env.observation_space.shape[0]
        try:
            self.size_out = _env.action_space.shape[0]
        except:
            self.size_out = _env.action_space.n
        if 'CarRacing' in args.env: self.size_out = 6
        self.gamma = args.gamma

    def forward(self, x):
        return None

    def normalized_init(self, weights, std=1.0):
        x = torch.randn(weights.size())
        x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
        return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=False):
        super(Conv, self).__init__()
        padding = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


def hogwild(model, args, train_func, test_func):
    # Hogwild algorithm
    model.share_memory()
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train_func, args=(model, args, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Test trained model
    test_func(model, args)


def init_hidden(batch_size, size_hidden):
    # Initialize hidden states for a LSTM cell
    hx = Variable(torch.zeros(batch_size, size_hidden))
    cx = Variable(torch.zeros(batch_size, size_hidden))
    return hx, cx


def plot(mean, std, args, labels=None, ylim_bot=None, save_path=None, walltime=None):
    # Plots the learning of a worker
    sns.set(style="darkgrid", font_scale=1.25)
    plt.figure(figsize=(12,10))

    if len(mean.shape) > 1 or walltime is not None:
        for i in range(len(mean)):
            if walltime is None:
                iterations = np.array(range(len(mean[i]))) * args.print_freq
            else:
                iterations = walltime[i] / 60
            plt.fill_between(iterations, mean[i]-std[i], mean[i]+std[i], alpha=0.2)
            label = labels[i] if labels is not None else None
            plt.plot(iterations, mean[i], label=f'A3C, {label}' if label is not None else 'A3C')
    else:
        iterations = np.array(range(len(mean))) * args.print_freq
        if std is not None:
                plt.fill_between(iterations, mean-std, mean+std, alpha=0.2)
        plt.plot(iterations, mean, label='A3C')

    plt.title(args.env)
    plt.xlabel('Iteration' if walltime is None else 'Walltime (minutes)')
    plt.ylabel('Mean reward')
    plt.legend()

    
    if len(iterations) > 1: plt.xlim(left=0, right=iterations[-1])
    if ylim_bot is not None: plt.ylim(bottom=ylim_bot)

    path = get_results_path() + 'reward.png' if save_path is None else save_path
    plt.savefig(path)
    plt.close()


def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def get_results_path(timestamp = dt):
    path = os.getcwd() + '/results/' + timestamp + '/'
    if not os.path.exists(path): os.makedirs(path)
    return path


def save_rewards(rewards):
    with open(get_results_path() + 'rewards.pkl', 'wb') as f:
        pickle.dump(rewards, f)


def save_walltime(walltime):
    with open(get_results_path() + 'walltime.pkl', 'wb') as f:
        pickle.dump(walltime, f)


def save_model(model, args, rewards):
    path = get_results_path()
    torch.save(model.state_dict(), path + 'model.pkl')
    with open(path + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)
    save_rewards(rewards)


def load_args(timestamp):
    path = get_results_path(timestamp)
    with open(path + 'args.pkl', 'rb') as f:
        return pickle.load(f)


def load_model(model, timestamp):
    path = get_results_path(timestamp)
    model.load_state_dict(torch.load(path + 'model.pkl'))
    return model