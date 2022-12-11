import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.replay_buffer import ReplayBuffer
import math


device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
class DQN(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=[64, 32, 32], action_input_layer=0,
                 output_layer_init_std=None, activation="relu", **kwargs):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size

        if hidden_layers:
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.layers.append(nn.Linear(hidden_layers[-1], self.action_size))

        # initialise layer weights
        for i in range(len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        else:
            nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.01

        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        Q = self.layers[0](x)
        for i in range(1, num_layers):
            if self.activation == "relu":
                Q = self.layers[i](F.relu((Q)))
            elif self.activation == "leaky_relu":
                Q = self.layers[i](F.leaky_relu((Q), negative_slope))
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=[64, 32],
                 output_layer_init_std=0.2, init_type="kaiming", activation="relu", init_std=None):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0

        # create layers
        self.layers = nn.ModuleList()
        input_size = self.state_size
        last_layer_size = input_size

        if hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            last_layer_size = hidden_layers[-1]

        #self.layers.append(nn.Linear(last_layer_size, self.action_parameter_size))
        self.output_layer = nn.Linear(last_layer_size, self.action_parameter_size)
        self.passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        for i in range(0, len(self.layers)):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        

        #if output_layer_init_std is not None:
         #   nn.init.normal_(self.output_layer.weight, std=1/math.sqrt(last_layer_size))
        #else:
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        nn.init.zeros_(self.passthrough_layer.weight)
        nn.init.zeros_(self.passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.passthrough_layer.requires_grad = False
        self.passthrough_layer.weight.requires_grad = False
        self.passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        negative_slope = 0.01
        num_layers = len(self.layers)
        
        if num_layers:
            x = self.layers[0](x)
            for i in range(1, num_layers):
                if self.activation == "relu":
                    x = self.layers[i](F.relu((x)))
                elif self.activation == "leaky_relu":
                    x = self.layers[i](F.leaky_relu((x), negative_slope))
                else:
                    raise ValueError("Unknown activation function "+str(self.activation))

        action_params = self.output_layer(x)
        action_params += self.passthrough_layer(state)
        
        action_params = torch.tanh(action_params)
        return action_params



def update_target_network(q_network, target_network, tau, soft=True):
    parameters_q = torch.nn.Module.state_dict(q_network)
    if not soft:
        torch.nn.Module.load_state_dict(target_network, parameters_q)
    else:
        #parameters_target = torch.nn.Module.state_dict(target_network)
        for target_param, param in zip(target_network.parameters(), q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class PDQNAgent(nn.Module):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """
    def __init__(self,
                 state_size,
                 action_space,
                 update_target_freq=50,
                 epsilon_initial=1,
                 epsilon_final=0.05,
                 epsilon_final_steps=10000,
                 batch_size=64,
                 discount_factor=0.99999,
                 tau_discrete=0.01,  # Polyak averaging factor for copying target weights
                 tau_param=0.001,
                 replay_buffer_size=1000000,
                 learning_rate_q=0.0001,
                 learning_rate_param=0.00001,
                 loss_func=torch.nn.MSELoss(), #F.mse_loss, # F.mse_loss
                 clip_grad=10, #10,
                 seed=None):
        super(PDQNAgent, self).__init__()

        self.state_size = state_size
        self.num_discrete_actions = action_space.spaces[0].n
        self.parameter_sizes = [ele.shape[0] for ele in list(action_space.spaces[1].spaces)]
        self.action_parameter_size = sum(self.parameter_sizes)

        #self.action_max = torch.from_numpy(np.ones((self.num_discrete_actions,))).float().to(device)
        #self.action_min = -self.action_max.detach()
        #self.action_range = (self.action_max-self.action_min).detach()

        print([action_space.spaces[1].spaces[i].high for i in range(self.num_discrete_actions)])
        self.action_parameter_max_numpy = np.concatenate([action_space.spaces[1].spaces[i].high for i in range(self.num_discrete_actions)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([action_space.spaces[1].spaces[i].low for i in range(self.num_discrete_actions)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_final_steps = epsilon_final_steps

        if seed:
            print("random seed: ", seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

        self.update_target_freq = update_target_freq

        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate_q = learning_rate_q
        self.learning_rate_param = learning_rate_param

        self.tau_discrete = tau_discrete
        self.tau_param = tau_param
        self.num_steps_taken = 0
        self.num_episodes = 0
        self.updates = 0
        self.clip_grad = clip_grad

        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_size, self.action_parameter_size + 1)

        self.q_network = DQN(self.state_size, self.num_discrete_actions, self.action_parameter_size).to(device)
        self.q_network_target = DQN(self.state_size, self.num_discrete_actions, self.action_parameter_size).to(device)
        self.q_network_losses = []
        update_target_network(self.q_network, self.q_network_target, tau=1, soft=False)
        self.q_network_target.eval() # target models are in eval mode

        self.param_actor = ParamActor(self.state_size, self.num_discrete_actions, self.action_parameter_size).to(device)
        self.param_actor_target = ParamActor(self.state_size, self.num_discrete_actions, self.action_parameter_size).to(device)
        self.param_actor_losses = []
        update_target_network(self.param_actor, self.param_actor_target, tau=1, soft=False)
        self.param_actor_target.eval()

        self.loss_func = loss_func

        self.dqn_optimiser = optim.Adam(self.q_network.parameters(), lr=self.learning_rate_q)
        self.param_optimiser = optim.Adam(self.param_actor.parameters(), lr=self.learning_rate_param)

    def start_episode(self):
        pass

    def update_epsilon(self):
        if self.num_episodes < self.epsilon_final_steps:
            decay = (self.epsilon_initial - self.epsilon_final) * self.num_episodes / self.epsilon_final_steps
            self.epsilon = self.epsilon_initial -  decay
        else:
            self.epsilon = self.epsilon_final

    def end_episode(self):
        self.num_episodes += 1
    
    def get_next_action(self, state, epsilon=None):
        if not epsilon:
            epsilon = self.epsilon

        with torch.no_grad():
            # epsilon greedy
            rnd = np.random.random()
            if rnd < epsilon:
                # explore
                discrete_action =  np.random.randint(self.num_discrete_actions)
                all_action_parameters = torch.from_numpy(np.random.uniform(-1, 1, 3))

            else:
                # exploit
                state = torch.from_numpy(state).to(device)
                all_action_parameters = self.param_actor.forward(state)
                
                # select maximum action
                q_values = self.q_network.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                q_values = q_values.detach().cpu().data.numpy()
                discrete_action = np.argmax(q_values)

            all_action_parameters = all_action_parameters.cpu().data.numpy()
            action_parameters = all_action_parameters[discrete_action: discrete_action + 1]
            
        return discrete_action, action_parameters, all_action_parameters


    def get_greedy_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(device)
            all_action_parameters = self.param_actor.forward(state)

            q_values = self.q_network.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
            q_values = q_values.detach().cpu().data.numpy()
            discrete_action = np.argmax(q_values)
            action_parameters = all_action_parameters[discrete_action: discrete_action + 1]

        return discrete_action, action_parameters, all_action_parameters

    def step(self, state, action, reward, next_state, terminal):

        self.num_steps_taken += 1
        
        # store transition in replay buffer
        self.replay_buffer.append_transition(state, np.concatenate(([action[0]],action[1])).ravel(), reward, next_state, terminal)
        
        if self.num_steps_taken >= self.batch_size:
            # sample from replay buffer and optimize networks
            losses = self.update_network()
            self.updates += 1

            # update weights for prioritzed experience replay
            self.replay_buffer.update_weights(losses)
        
        # update target network every update_target_freq steps
        if self.num_steps_taken >= self.batch_size and ((self.num_steps_taken - self.batch_size) % self.update_target_freq) == 0:
            update_target_network(self.q_network, self.q_network_target, self.tau_discrete, soft=False)
            update_target_network(self.param_actor, self.param_actor_target, self.tau_param, soft=False)

            
    def update_network(self):
        # calculate td loss and update network weights

        # only start once steps is greater than batch size
        if self.num_steps_taken < self.batch_size:
            return

        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).to(device)
        actions_and_params = torch.from_numpy(actions).to(device)
        actions = actions_and_params[:, 0].long()
        action_parameters = actions_and_params[:, 1:] # [64, 3]
        rewards = torch.from_numpy(rewards).to(device).squeeze()
        next_states = torch.from_numpy(next_states).to(device)
        terminals = torch.from_numpy(terminals).to(device).squeeze()

        # optimize action parameter network
        with torch.no_grad():
            action_params = self.param_actor(states)
        action_params.requires_grad = True
        q_values = self.q_network.forward(states, action_params)
        param_loss = - q_values.sum(dim=-1).mean()
        self.param_actor_losses.append(param_loss.item())

        self.param_optimiser.zero_grad()
        param_loss.backward()
        self.param_optimiser.step()


        # optimize Q-network 
        with torch.no_grad():
            # Pedict future Q using target network
            pred_next_action_parameters = self.param_actor_target.forward(next_states)
            pred_next_q_values = self.q_network_target(next_states, pred_next_action_parameters)
            
            # max future reward
            pred_future_return = torch.max(pred_next_q_values, 1, keepdim=True)[0].squeeze()

            # target prediction for q value
            target_predictions = rewards + (1 - terminals) * self.discount_factor * pred_future_return

        # Compute current Q-values using policy network
        predictions = self.q_network(states, action_parameters)
        predictions = predictions.gather(1, actions.view(-1, 1)).squeeze() # q values for the action taken
        
        #losses for each of batch samples
        losses =  abs(predictions - target_predictions)
        # loss (MSE) of the DQN
        dqn_loss = self.loss_func(predictions, target_predictions)
        self.q_network_losses.append(dqn_loss.item())
        self.dqn_optimiser.zero_grad()
        dqn_loss.backward()
        #if self.clip_grad > 0:
        #    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clip_grad)
        self.dqn_optimiser.step()

        return losses

        
    def save_models(self, dir):
        torch.save(self.q_network.state_dict(), dir + '/q_network.pt')
        torch.save(self.param_actor.state_dict(), dir + '/param_actor.pt')


    def load_models(self, dir):
        self.q_network.load_state_dict(torch.load(dir + '/q_network.pt', map_location='cpu'))
        self.param_actor.load_state_dict(torch.load(dir + '/param_actor.pt', map_location='cpu'))
