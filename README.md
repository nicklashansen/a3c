# Asynchronous Advantage Actor-Critic (A3C) using Generalized Advantage Estimation (PyTorch)

A3C utilizes distributed learning where a number of workers interact with the environment and update model parameters asynchronously, which effectively removes the need for a memory buffer due to high data throughput. Another advantage of
distributed learning is that it allows for parallel generation of multiple rollouts, which often is a performance bottleneck in
training of RL systems.

As rollouts are generated on-policy, however, it is likely that all trajectories will be similar as action probabilities gradually become near-zero for all but one action in the discrete case, which ultimately limits exploration. A3C addresses this
problem by introducing an entropy-term to the loss function, which is discussed in section 2. We extend the A3C by replac-
ing the advantage estimator used in (Minh, 2016) by the Generalized Advantage Estimate (GAE) as proposed by (Schulman, 2015), and evaluate the algorithm on a number of environments.

See our paper for more details.
