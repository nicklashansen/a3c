import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from utils import get_results_path, plot, load_args

'''
@author:
Nicklas Hansen

Plot a learning curve with:
python plot.py FOLDER1&FOLDER2
where FOLDER1 and FOLDER2 are the names of auto-generated folders in /results.
'''

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--labels', type=str, default='')
    parser.add_argument('--ylim_bot', type=float, default=None)
    parser.add_argument('--print_args', type=bool, default=False)
    parser.add_argument('--walltime', type=bool, default=False)
    parser.add_argument('--path', type=str, default='output-plot.png')
    args = parser.parse_args()

    # Get file paths
    data = args.data.split('&')
    labels = None if len(args.labels) == 0 else args.labels.split('&')
    if labels is not None and len(data) != len(labels):
        raise ValueError('Number of labels must match number of sequences!')

    # Load data
    def get_rewards(data):
        for d in data:
            with open(get_results_path(d) + 'rewards.pkl', 'rb') as f:
                yield pickle.load(f)
                
    # Load args
    args2 = load_args(data[0])
    if args.print_args is True:
        print(args2)

    # Load walltime
    def get_walltime(data):
        for d in data:
            with open(get_results_path(d) + 'walltime.pkl', 'rb') as f:
                yield pickle.load(f)

    rewards = list(get_rewards(data))
    walltime = list(get_walltime(data)) if args.walltime is True else None
    running_mean, running_std = [], []
    running_walltime = []

    for j in range(len(rewards)):

        train_rewards = rewards[j]
        train_walltime = walltime[j] if walltime is not None else None
        mean, std = [], []
        wt = []

        for i in range(0, len(train_rewards), args2.print_freq):
            if i == 0: continue
            recent_rewards = train_rewards[i-args2.print_freq:i]
            mean.append(np.mean(recent_rewards))
            std.append(np.std(recent_rewards))
            if walltime is not None: wt.append(train_walltime[i])

        running_mean.append(np.array(mean))
        running_std.append(np.array(std))
        running_walltime.append(np.array(wt))

    plot(np.array(running_mean), np.array(running_std), args2, labels, ylim_bot=args.ylim_bot, save_path=os.getcwd() + '/results/' + args.path, walltime=np.array(running_walltime) if args.walltime is True else None)


