
import torch.nn as nn
import numpy as np


class ReplayBuffer(nn.Module):

    def __init__(self, buffer_size, state_size, action_shape, batch_size=64, alpha=0.7, epsilon_probabilities=1*1e-6):
        self.max_size = buffer_size
        self.batch_size = batch_size

        self.counter = 0 #to do buffer updates when there is overflow
        self.weights = np.zeros(self.max_size) #weights for each transitino to update the probabilities
        self.epsilon_probabilities = epsilon_probabilities #min probability for each sample to be taken for the batch

        self.counter = 0
        self.alpha = alpha
        self.size = 0
        self.start_idx = 0

        self.states =  Buffer(self.max_size, (state_size,))
        self.actions = Buffer(self.max_size, (action_shape,))
        self.rewards = Buffer(self.max_size, (1,))
        self.next_states = Buffer(self.max_size, (state_size,))
        #self.next_actions = collections.deque([], self.max_len_buffer) if next_actions else None
        self.terminals = Buffer(self.max_size, (1,))

    #Adds each transition to the buffer
    def append_transition(self, state, action, reward, next_state, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(terminal)

        #Initialiase all the weights to be equal for the first batch
        if self.size <= self.batch_size:
            self.weights[self.start_idx] = 1/self.batch_size
        #Initialise the new weights to the max of exisiting buffer
        else:
            self.weights[self.start_idx] = np.max(self.weights)

        self.start_idx = (self.start_idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    #Sample the batch according to probabilities based on the weigths
    #which are based on the losses
    def sample(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size

        self.weights[:self.size] = self.weights[:self.size] + self.epsilon_probabilities
        #getting probabilities based on weights
        probabilities = self.weights**self.alpha/np.sum(self.weights**self.alpha)
        #sampling according to the probabilities

        self.batch_indices = np.random.choice(range(self.size), batch_size, True, probabilities[:self.size])
        
        states =  self.states.sample(self.batch_indices)
        actions =  self.actions.sample(self.batch_indices)
        rewards =  self.rewards.sample(self.batch_indices)
        next_states =  self.next_states.sample(self.batch_indices)
        terminals =  self.terminals.sample(self.batch_indices)

        return states, actions, rewards, next_states, terminals

    #Update the weights based on the losses which have just been updated in the
    #network optimisation
    def update_weights(self, losses):
        self.weights[self.batch_indices] = losses.detach().numpy().reshape(self.batch_indices.size,)


class Buffer():
    def __init__(self, buffer_size, state_shape, dtype='float32'):
        self.max_size = buffer_size

        self.buffer = np.zeros((self.max_size,) + state_shape).astype(dtype)

        self.size = 0
        self.start_idx = 0


    #Adds each transition to the buffer
    def append(self, ele):
        if self.size < self.max_size:
            self.size += 1
        elif self.size == self.max_size:
            self.start_idx = (self.start_idx + 1) % self.max_size
        else:
            raise RuntimeError()

        self.buffer[(self.start_idx + self.size + 1) % self.max_size] = ele

    def sample(self, idxs):
        return self.buffer[(self.start_idx + idxs) % self.max_size]

