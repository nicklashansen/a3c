# Asynchronous Advantage Actor-Critic (A3C) using Generalized Advantage Estimation (PyTorch)

### What is A3C?
A3C utilizes distributed learning where a number of workers interact with the environment and update model parameters asynchronously, which effectively removes the need for a memory buffer due to high data throughput. Another advantage of
distributed learning is that it allows for parallel generation of multiple rollouts, which often is a performance bottleneck in
training of RL systems.

As rollouts are generated on-policy, however, it is likely that all trajectories will be similar as action probabilities gradually become near-zero for all but one action in the discrete case, which ultimately limits exploration. A3C addresses this
problem by introducing an entropy-term to the loss function, which is discussed in section 2. We extend the A3C by replacing the advantage estimator used in (Minh, 2016) by the Generalized Advantage Estimate (GAE) as proposed by (Schulman, 2015), and evaluate the algorithm on a number of environments.

### Experiments

CartPole

![CartPole](https://i.imgur.com/B3t0Gjh.gif)

MountainCar

![MountainCar](https://i.imgur.com/UICkgp0.gif)

LunarLander

![LunarLander](https://i.imgur.com/1JkUazV.gif)

### Additional experiments

Implementation also includes code for the CarRacing and Atari environments (ConvNets and pre-processing), but those require significantly more resources to converge to a nice solution and are thus not included here.

It is also possible to extend A3C to continuous action spaces. For more information on that, visit https://github.com/dgriff777/a3c_continuous.

### Results

We investigated the impact of exploration by an added entropy term to the loss function and found that it improves exploration substantially for the MountainCar problem:

![Exploration](https://i.imgur.com/79cUyvq.png)

See our paper/report for details as well as more experiments.

### Questions?

You are very welcome to read our paper/report for more information and you are also free to inspect and use the code in our repository.
If you have any questions that are left unanswered, you can contact me on the following email: hello@nicklashansen.com.
