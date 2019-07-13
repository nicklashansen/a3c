import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from gym.wrappers import Monitor
import time
import argparse
import sys
import matplotlib.pyplot as plt
from utils import *

'''
@authors:
Nicklas Hansen,
Peter Ebert Christensen

Run the algorithm with:
python a3c.py
'''

class AC(Agent):
    # Agent to be used in 1D environments
    def __init__(self, args):
        super(AC, self).__init__(args)
        self.fc = nn.Linear(self.size_in, args.size_hidden)
        self.fc2 = nn.Linear(args.size_hidden, args.size_hidden)
        self.fc_policy = nn.Linear(args.size_hidden, self.size_out)
        self.fc_value = nn.Linear(args.size_hidden, 1)
        self.lstm = nn.LSTMCell(args.size_hidden, args.size_hidden)
        self.init_weights_biases()
        
    def forward(self, x, hx, cx):
        x = Tensor(x).unsqueeze(0)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.fc_policy(x), self.fc_value(x), hx, cx

    def init_weights_biases(self):
        self.fc_policy.weight.data = self.normalized_init(self.fc_policy.weight.data, std=0.01)
        self.fc_value.weight.data = self.normalized_init(self.fc_value.weight.data, std=1.00)
        self.fc.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc_policy.bias.data.fill_(0)
        self.fc_value.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)


class ConvAC(AC):
    # Agent to be used in 2D environments
    def __init__(self, args, num_filters1=16, num_filters2=32, num_filters3=16):
        super(ConvAC, self).__init__(args)
        self.fc = None
        self.conv1 = Conv(1, num_filters1, kernel_size=8, stride=4)
        self.conv2 = Conv(num_filters1, num_filters2, kernel_size=4, stride=2)
        self.conv3 = Conv(num_filters2, num_filters3, kernel_size=3, stride=1)

        lstm_in = 2 ** 2 if 'CarRacing' in args.env else 6 ** 2
        self.lstm = nn.LSTMCell(num_filters3*lstm_in, args.size_hidden)

    def forward(self, x, hx, cx):
        x = self.preprocess(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x,(hx,cx))
        return self.fc_policy(hx), self.fc_value(cx), hx, cx

    def preprocess(self, x):
        # Determine if we are playing Atari
        atari = len(x.shape) == 3 and x.shape[0] == 210 and x.shape[1] == 160

        if atari is True:
            # Permute dimensions
            x = Variable(Tensor(x)).unsqueeze(0).permute([0, 3, 2, 1]).data.numpy()

            # Downsample image
            x = x[:, :, ::2, ::2]

            # Convert to grayscale and percentages
            x = np.mean(x, axis=1) / 255

            # Crop image and reshape
            x = Tensor(x[:, :, 17:105-8].reshape(-1, 1, 80, 80))

            # Subtract min value
            x = x - torch.min(x)

        else:
            # Convert to grayscale
            x = np.dot(x[..., :], [0.299, 0.587, 0.114])

            # Downsample image
            x = x[::2, ::2]

            # Normalize and unsqueeze
            x = Tensor(x / 128 - 1).unsqueeze(0).unsqueeze(0)

        return x
        

def train(model, args, rank):
    # Initialize environment and optimizer
    env = gym.make(args.env)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize variables for performance tracking
    train_reward, best_reward = [], -sys.maxsize
    train_walltime, time_start = [], time.time()
    running_mean, running_std = [], []

    for ep in range(args.episodes):

        s = env.reset()
        step, done, ep_reward = 0, False, []
        hx, cx = init_hidden(1, args.size_hidden)

        if 'MountainCar' in args.env and ep > 1000:
            args.beta = 0

        while not done:

            transitions = []
            hx = Variable(hx.data)
            cx = Variable(cx.data)

            for _ in range(args.lag):
                
                # Determine which action to take
                logit,value, hx, cx = model.forward(s, hx, cx)
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(-1)
                action = prob.multinomial(1).data
                log_prob = log_prob.gather(1, Variable(action))

                # Take a step in environment
                if 'MountainCar' in args.env:
                    for _ in range(4):
                        s, r, done, _ = env.step(action.squeeze().numpy())
                        if done: break
                    r *= 4
                elif 'CarRacing' in args.env:
                    controls = {
                        0: [-0.25, 0.25, 0],
                        1: [0.25, 0.25, 0],
                        2: [0, 0.25, 0],
                        3: [-0.25, 0, 0.25],
                        4: [0.25, 0, 0.25],
                        5: [0, 0, 0.25]
                    }
                    action, r = np.array(controls[int(action.squeeze().numpy())]), 0
                    for _ in range(4):
                        if rank == 0 and ep % 5 == 0: env.render()
                        s, r_step, done, _ = env.step(action)
                        r += r_step
                        if done: break

                else:
                    s, r, done, _ = env.step(action.squeeze().numpy())
                transitions.append((entropy, s, r/100, value, log_prob))
                ep_reward.append(r)
                
                step += 1
                if done: break
            
            # Unzip transitions and initialize variables for update
            entropies, states, rewards, values, log_probs = zip(*transitions)
            R = torch.zeros(1,1)
            
            if not done:
                _, value, _, _= model.forward(states[-1], hx, cx)
                R = value.data
                
            policy_loss, value_loss, gae = 0, 0, torch.zeros(1,1)

            values = list(values)
            values.append(Variable(R))
            R = Variable(R)
            
            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)
                
                # Compute TD-error
                delta_t = rewards[i] + args.gamma * values[i+1].data - values[i].data
                
                # Generalized Advantage Estimation
                gae = gae * args.gamma * args.Lambda + delta_t
                
                # Compute policy loss including entropy
                policy_loss = policy_loss - log_probs[i] * Variable(gae) - args.beta * entropies[i]
                
            # Update global network
            opt.zero_grad()
            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            #clip_grad_norm_(model.parameters(),args.max_grad_norm)
            opt.step()

        train_reward.append(np.sum(ep_reward))
        train_walltime.append(time.time() - time_start)

        # Track model performance
        if rank == 0 and ep % args.print_freq == 0 and ep > 0:
            most_recent_reward = train_reward[-1]
            recent_rewards = train_reward[-args.print_freq:]
            mean, std = np.mean(recent_rewards), np.std(recent_rewards)
            print('Episode {:4d} | Mean reward: {:.2f} | Std reward: {:.2f}'.format(ep, mean, std))
            running_mean.append(mean)
            running_std.append(std)
            plot(np.array(running_mean), np.array(running_std), args)
            save_rewards(train_reward)
            save_walltime(train_walltime)

            # Save model if it improved
            if most_recent_reward >= best_reward:
                best_reward = most_recent_reward
                save_model(model, args, train_reward)

    # Close environment when done
    env.close()


def test(model, args, verbose=True):
    # Initialize environment and model
    env = Monitor(gym.make(args.env), './recordings', force=True)
    model.eval()
    
    # Initialize variables
    done, ep_reward = False, []
    s = env.reset()
    hx, cx = init_hidden(1, args.size_hidden)

    # Generate rollout
    while not done: # and step < env.spec.timestep_limit:

        # Render if enabled
        if args.render: env.render()

        # Take a step in environment
        logit, _, _, _ = model.forward(s, hx, cx)
        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(1).data
        s, r, done, _ = env.step(action.squeeze().numpy())
        ep_reward.append(r)

        if done: break

    # Close environment and show performance
    env.close()
    if verbose is True:
        print('Test agent achieved a reward of', np.sum(ep_reward))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CarRacing-v0')
    parser.add_argument('--size_hidden', type=int, default=256)
    parser.add_argument('--num_processes', type=int, default=3)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--lag', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--Lambda', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--render', type=bool, default=True)
    args = parser.parse_args()

    # Initialize model
    gym.logger.set_level(40)
    _env = gym.make(args.env)
    args.conv = len(_env.observation_space.shape) == 3
    model = ConvAC(args) if args.conv == True else AC(args)
    print('Initializing training...')

    # Run
    hogwild(model, args, train, test)

