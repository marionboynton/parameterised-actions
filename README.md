# parameterised-actions
Reinforcement Learning for Parameterised Action Spaces

The Platform Environment: https://github.com/cycraig/gym-platform

The platform environment tests agents in a parameterised action space - discrete actions with continuous parameters. At each step the agent selects the discrete action to take (run, leap, hop) as well as the parameters to use with that action (distance to cover).

This Repository includes an implementation of the Parameterized Deep Q network (PDQN) to learn in this domain as well as Hybrid Proximal Policy Optimization model as described in [this paper](https://arxiv.org/pdf/1903.01344.pdf), which maintains 2 policy actors - one for the continuous actions and one forthe discrete actions - and a single critic network that approximates the state values.

### Dependencies
- Python 3.6
- gym 0.10.5
- pygame 1.9.4
- numpy
- gym-platform 0.0.1


### example usage
`python train_and_test_hppo.py`


### References:
- Reinforcement Learning with Parameterized Actions: https://arxiv.org/pdf/1509.01644.pdf
- Parametrized Deep Q-Networks Learning: https://arxiv.org/pdf/1810.06394.pdf
- an implementation of PDQN: https://github.com/cycraig/MP-DQN/blob/master/agents/pdqn.py
- Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space: https://arxiv.org/pdf/1903.01344.pdf
- Proximal Policy Optimization Algorithms: https://arxiv.org/pdf/1707.06347.pdf
