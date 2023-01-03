import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, MultivariateNormal
from scipy import stats

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class ActorDiscrete(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=[64, 32, 32], output_layer_init_std=0.2):
        super(ActorDiscrete, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # create layers
        self.layers = nn.ModuleList()

        if hidden_layers:
            self.layers.append(nn.Linear(self.state_dim, hidden_layers[0]))
            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.Linear(hidden_layers[-1], self.action_dim))
        else:
            self.layers.append(nn.Linear(self.state_dim, self.action_dim))

        # initialise layer weights
        for i in range(len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity="tanh")
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        else:
            nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state):
        # implement forward
        num_layers = len(self.layers)
        a = self.layers[0](state)
        for i in range(1, num_layers):
            a = self.layers[i](torch.tanh(a))
            
        a = F.softmax(a, dim=-1)
        return a

    def get_action(self, state, deterministic=False, explore=False):
        action_probs = self.forward(state)
        if deterministic:
            return torch.argmax(action_probs)
        
        distribution = Categorical(action_probs)
        if explore:
            action = torch.tensor(np.random.randint(0, self.action_dim))
        else:
            action = distribution.sample()
        action_logprobs = distribution.log_prob(action)

        return action, action_logprobs
    
    def evaluate(self, state, action):
         action_probs = self.forward(state)
         distribution = Categorical(action_probs)

         action_logprobs = distribution.log_prob(action)
         distribution_entropy = distribution.entropy()

         return action_logprobs, distribution_entropy


class ActorContinuous(nn.Module):

    def __init__(self, state_dim, action_dim, action_std_init, hidden_layers=[64, 32, 32], output_layer_init_std=0.2):
        super(ActorContinuous, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim # 3

        self.action_variance = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # create layers
        self.layers = nn.ModuleList()

        if hidden_layers:
            self.layers.append(nn.Linear(self.state_dim, hidden_layers[0]))
            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.Linear(hidden_layers[-1], self.action_dim))
        else:
            self.layers.append(nn.Linear(self.state_dim, self.action_dim))

        # initialise layer weights
        for i in range(len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity="tanh")
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        else:
            nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state):
        # implement forward
        negative_slope = 0.01

        num_layers = len(self.layers)
        a = self.layers[0](state)
        for i in range(1, num_layers):
            a = self.layers[i](torch.tanh(a))
        return a

    def set_action_std(self, new_action_std):
        self.action_variance = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def get_action(self, state, deterministic=False, explore=False):
         action_avg = self.forward(state)
         if deterministic:
            return action_avg

         variance = self.action_variance.expand_as(action_avg).type(torch.float32)
         cov_mat = torch.diag_embed(variance.type(torch.float32)).to(device)
         distribution = MultivariateNormal(action_avg, cov_mat)

         if explore:
            action = np.random.uniform(-1, 1, self.action_dim) # random dxs in range [-1, 1)
            action = torch.from_numpy(action).float()
         else:
            action = distribution.sample()

         log_prob = distribution.log_prob(action)
         return action, log_prob
    
    def evaluate(self, state, action):
         action_avg = self.forward(state) # [50, 3]
        
         action_variance = self.action_variance.expand_as(action_avg).type(torch.float32) # expand to state tensor dimensions
         cov_mat = torch.diag_embed(action_variance).to(device)
         distribution = MultivariateNormal(action_avg, cov_mat)

         action_logprobs = distribution.log_prob(action)
         distribution_entropy = distribution.entropy()

         return action_logprobs, distribution_entropy


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 32, 32], activation="relu"):
        super(Critic, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim

        # create layers
        self.layers = nn.ModuleList()

        if hidden_layers:
            self.layers.append(nn.Linear(self.state_dim, hidden_layers[0]))
            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.Linear(hidden_layers[-1], self.action_dim))
        else:
            self.layers.append(nn.Linear(self.state_dim, self.action_dim))
        
         # initialise layer weights
        for i in range(len(self.layers)):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, state):
        # implement forward
        num_layers = len(self.layers)
        v = self.layers[0](state)
        for i in range(1, num_layers):
            v = self.layers[i](F.relu(v))
            
        return v


class HPPO(nn.Module):
    """
    Hybrid PPO agent for parameterised action spaces
    """
    def __init__(self,
                 state_size,
                 action_space,
                 action_std_init, # starting std for action distribution
                 eps_clip=0.2,
                 num_epochs=50, # run policy for k epochs on stored transitions and optimize
                 discount_factor=0.9999, # trying disocunt factor = 1 so the agent prioritises th
                 lr_discrete=0.0005,
                 lr_continuous=0.0001,
                 lr_critic = 0.0002,
                 loss_func=torch.nn.MSELoss(), #F.mse_loss, # F.mse_loss
                 vf_coef = 0.5,
                 ent_coef_discrete = 0.05,
                 ent_coef_cont = 0.05,
                 seed=None):
        super(HPPO, self).__init__()
        
        self.num_discrete_actions = action_space.spaces[0].n
        self.state_size = state_size

        self.cont_action_dim = len(action_space.spaces[1].spaces) # 3
        self.parameter_sizes = [ele.shape[0] for ele in list(action_space.spaces[1].spaces)]
        self.action_parameter_size = sum(self.parameter_sizes)

        self.action_parameter_max_numpy = np.concatenate([action_space.spaces[1].spaces[i].high for i in range(self.num_discrete_actions)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([action_space.spaces[1].spaces[i].low for i in range(self.num_discrete_actions)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

        self.eps_clip = eps_clip
        self.num_epochs = num_epochs
        self.action_std = action_std_init
        self.discount_factor = discount_factor
        
        self.lr_discrete = lr_discrete
        self.lr_continuous = lr_continuous
        self.lr_critic = lr_critic

        self.num_steps_taken = 0
        self.num_episodes = 0

        if seed:
            print("random seed: ", seed)
            torch.manual_seed(seed)
            np.random.seed(seed)


        self.buffer = Buffer()

        self.critic = Critic(self.state_size, 1)
        self.vf_coef = vf_coef
        self.ent_coef_discrete = ent_coef_discrete
        self.ent_coef_cont = ent_coef_cont
  
        self.discrete_policy = ActorDiscrete(self.state_size, self.num_discrete_actions).to(device)
        self.old_discrete_policy = ActorDiscrete(self.state_size, self.num_discrete_actions).to(device)
        self.old_discrete_policy.load_state_dict(self.discrete_policy.state_dict())

        self.cont_policy = ActorContinuous(self.state_size, self.cont_action_dim, self.action_std).to(device)
        self.old_cont_policy = ActorContinuous(self.state_size, self.num_discrete_actions, self.action_std).to(device)
        self.old_cont_policy.load_state_dict(self.cont_policy.state_dict())
        self.action_std = action_std_init

        self.optimizer = optim.Adam([{'params': self.discrete_policy.parameters(), 'lr': self.lr_discrete}, {'params': self.cont_policy.parameters(), 'lr': self.lr_continuous}, {'params': self.critic.parameters(), 'lr': self.lr_critic}]) #, betas=(0.95, 0.999))
        self.critic_losses = []

        self.loss_func = loss_func

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.cont_policy.set_action_std(new_action_std)
        self.old_cont_policy.set_action_std(new_action_std)

    def decay_action_std(self, decay_rate, min_std):
        self.action_std = self.action_std - decay_rate
        self.action_std = max(min_std, round(self.action_std, 4))
        if self.action_std == min_std:
            print(f"setting actor output action std to min std: {min_std}")
        else:
            print(f"setting actor output action std to {self.action_std}")
        self.set_action_std(self.action_std)


    def decay_entropy_coeff(self, decay_rate, min_coeff, coefficient=None):
        if not coefficient:
            self.ent_coef_discrete = self.ent_coef_discrete - decay_rate
            self.ent_coef_cont = self.ent_coef_cont - decay_rate
            self.ent_coef_discrete = max(min_coeff, round(self.ent_coef_discrete, 4))
            self.ent_coef_cont = max(min_coeff, round(self.ent_coef_cont, 4))

        else:
            coefficient = coefficient - decay_rate
            coefficient = max(min_coeff, round(self.coefficient, 4))

    def start_episode(self):
        pass

    def end_episode(self):
        self.num_episodes += 1


    def params_scaled(self, all_action_parameters):
        # scale params (distsance to move) to pass to environment
        all_action_parameters = torch.add(all_action_parameters, 1)
        all_action_parameters = torch.mul(all_action_parameters, torch.tensor(self.action_parameter_max_numpy))
        return all_action_parameters


    def get_discrete_action(self, state, deterministic=False, explore=False):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            discrete_action, log_prob = self.old_discrete_policy.get_action(state, deterministic, explore)
            
        return discrete_action.detach(), log_prob.detach()
    
    # deterministic discrete action
    def get_discrete_det_action(self, state):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            discrete_action = self.old_discrete_policy.get_action(state, deterministic=True)
            
        return discrete_action.detach()

    def get_continuous_action(self, state, deterministic=False, explore=False):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            cont_action, cont_log_prob = self.old_cont_policy.get_action(state, deterministic, explore)
            
        return cont_action.detach(), cont_log_prob.detach()
    
    # deterministic continuous action (dx parameter)
    def get_continuous_det_action(self, state):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            cont_action = self.old_cont_policy.get_action(state, deterministic=True)
            
        return cont_action.detach()
            
    def update_network(self):
        # Monte Carlo estimate of returns
        n = len(self.buffer.rewards)
        returns = [0]*n
        i = n - 1
        discounted_reward = 0
        for reward, terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.discount_factor * discounted_reward)
            returns[i] = discounted_reward
            i -= 1

        # normalize returns
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # create tensors from buffer transitions
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_discrete_actions = torch.squeeze(torch.stack(self.buffer.discrete_actions, dim=0)).detach().to(device)
        old_discrete_logprobs = torch.squeeze(torch.stack(self.buffer.discrete_logprobs, dim=0)).detach().to(device)
        old_cont_actions = torch.squeeze(torch.stack(self.buffer.cont_actions, dim=0)).detach().to(device)
        old_cont_logprobs = torch.squeeze(torch.stack(self.buffer.cont_logprobs, dim=0)).detach().to(device)

        discrete_losses = []
        cont_losses = []
        critic_losses = []

        # optimize policy for num epochs
        for _ in range(self.num_epochs):
            # DISCRETE POLICY
            # Evaluating old actions and values
            discrete_logprobs, discrete_dist_entropy = self.discrete_policy.evaluate(old_states, old_discrete_actions)
            state_values = self.critic.forward(old_states) # [50, 1]

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values) # [50]

            # Finding the ratio (pi_theta / pi_theta__old)
            discrete_ratios = torch.exp(discrete_logprobs - old_discrete_logprobs.detach()) # [50]

            # Finding Surrogate Loss
            advantages = returns - state_values.detach() # [50]

            surr = discrete_ratios * advantages
            surr_clipped = torch.clamp(discrete_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # final loss of clipped objective PPO
            discrete_loss = torch.min(surr, surr_clipped) + self.ent_coef_discrete*discrete_dist_entropy
            discrete_losses.append(discrete_loss.mean().item())

            # CONTINUOUS POLICY
            cont_logprobs, cont_dist_entropy = self.cont_policy.evaluate(old_states, old_cont_actions)
            cont_ratios = torch.exp(cont_logprobs - old_cont_logprobs.detach())
            surr = cont_ratios * advantages
            surr_clipped = torch.clamp(cont_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            cont_loss = torch.min(surr, surr_clipped) + self.ent_coef_cont*cont_dist_entropy
            cont_losses.append(cont_loss.mean().item())

            # CRITIC
            critic_loss = self.vf_coef*(self.loss_func(state_values, returns))
            critic_losses.append(critic_loss.mean().item())

            total_loss = - discrete_loss - cont_loss + critic_loss

            # gradient ascent / descent
            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.old_discrete_policy.load_state_dict(self.discrete_policy.state_dict())
        self.old_cont_policy.load_state_dict(self.cont_policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return stats.sem(discrete_losses), np.mean(discrete_losses), stats.sem(cont_losses), np.mean(cont_losses), stats.sem(critic_losses), np.mean(critic_losses)

        
    def save_models(self, dir):
        torch.save(self.old_discrete_policy.state_dict(), dir + '/discrete_policy.pt')
        torch.save(self.old_cont_policy.state_dict(), dir + '/cont_policy.pt')


    def load_models(self, dir):
        self.old_discrete_policy.load_state_dict(torch.load(dir + '/discrete_policy.pt', map_location='cpu'))
        self.discrete_policy.load_state_dict(torch.load(dir + '/discrete_policy.pt', map_location='cpu'))
        self.old_cont_policy.load_state_dict(torch.load(dir + '/cont_policy.pt', map_location='cpu'))
        self.cont_policy.load_state_dict(torch.load(dir + '/cont_policy.pt', map_location='cpu'))



class Buffer():
    def __init__(self):
        self.discrete_actions = []
        self.cont_actions = []
        self.states = []
        self.discrete_logprobs = []
        self.cont_logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def append_transition(self, discrete_action, cont_action, state, discrete_logprobs, cont_log_probs, reward, terminal):

        self.discrete_actions.append(discrete_action)
        self.cont_actions.append(cont_action)
        self.states.append(state)
        self.discrete_logprobs.append(discrete_logprobs)
        self.cont_logprobs.append(cont_log_probs)
        self.rewards.append(reward)
        self.is_terminals.append(terminal)
    
    def clear(self):
        self.discrete_actions = []
        self.cont_actions = []
        self.states = []
        self.discrete_logprobs = []
        self.cont_logprobs = []
        self.rewards = []
        self.is_terminals = []