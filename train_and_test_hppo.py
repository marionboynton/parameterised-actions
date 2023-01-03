import time
import numpy as np
import torch

import gym
import gym_platform
import os
from scipy import stats
from agents.hybrid_ppo import HPPO


# folder to save results
test_num = 1
results_path = f"hppo_results/{test_num}"
os.makedirs(results_path, exist_ok=True)

random_seed = 42
only_test_model = False
num_states = 4 # reduce to first 4 elements of observation as platform features are fixed

PARAMETERS_MIN = np.array([0, 0, 0])
PARAMETERS_MAX = np.array([
    30,  # run
    720,  # hop
    430  # leap
])

# HYPERPARAMETERS
# continuous action variance
action_std_init = 0.6
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = 5000  # action_std decay frequency (in num timesteps)

entropy_decay_freq = 5000
entropy_decay_rate = 0.01
min_entropy_coeff = 0.01

max_episodes = 10000
update_freq = 64


device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def convert_action(act_idx, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act_idx][:] = PARAMETERS_MAX[act_idx] * (act_param + 1)/2
    return (act_idx, params)

def reduce_state(state):
    return (state[:num_states])


def train(env, agent):
    start_time = time.time()

    total_reward = 0
    returns = []
    eval_returns = [] # running detemrinistic policy to evaluate
    
    max_episode_len = 30
    num_episodes = 0
    
    step = 0

    discrete_losses = [] # for plotting losses
    cont_losses = []
    critic_losses = []

    entropy_coeff = []
    action_std = []

    for _ in range(max_episodes):
        state, _ = env.reset()
        state = np.array(reduce_state(state), dtype=np.float32, copy=False)

        episode_reward = 0
        episode_steps = 0

        for _ in range(max_episode_len):
            cont_action, cont_log_prob = agent.get_continuous_action(state) # (1, 3)
            discrete_action, discrete_log_prob = agent.get_discrete_action(state)

            action = convert_action(discrete_action.item(), np.array(cont_action, dtype=np.float32, copy=False)[0])

            (next_state, steps), reward, terminal, _ = env.step(action)
            next_state = np.array(reduce_state(next_state), dtype=np.float32, copy=False)

            state = torch.from_numpy(state).float().to(device)
            reward = torch.from_numpy(np.asarray(reward)).float().to(device)
            terminals = torch.from_numpy(np.asarray(terminal)).float().to(device)

            agent.buffer.append_transition(discrete_action, cont_action, state, discrete_log_prob, cont_log_prob, reward, terminals)

            step += 1
            episode_steps += 1
            episode_reward += reward

            if step % update_freq == 0:
                discrete_loss_std, discrete_loss, cont_loss_std, cont_loss, critic_loss_std, critic_loss = agent.update_network()
                discrete_losses.append([discrete_loss, discrete_loss_std])
                cont_losses.append([cont_loss, cont_loss_std])
                critic_losses.append([critic_loss_std, critic_loss])

            
            if step % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            if step % entropy_decay_freq == 0:
                agent.decay_entropy_coeff(entropy_decay_rate, min_entropy_coeff)

            state = next_state
            
            if terminal:
                break

        num_episodes += 1
        returns.append(episode_reward)
        total_reward += episode_reward

        if num_episodes % 100 == 0:
            print(f'episode {num_episodes} episode reward: {episode_reward} r100: {np.mean(returns[-100:])} continuous action std dev: {agent.action_std} entropy coefficient: {agent.ent_coef_discrete}') #
            entropy_coeff.append(agent.ent_coef_discrete)
            action_std.append(agent.action_std)


        if num_episodes % 500 == 0:
            evals = []
            for _ in range(20):
                r = test(env, agent, deterministic=True)
                evals.append(r)
            eval_returns.append([np.mean(evals), max(evals), min(evals)])


    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))

    np.save(f'{results_path}/returns.npy', np.array(returns))
    np.save(f'{results_path}/eval_returns.npy', np.array(eval_returns))
    np.save(f'{results_path}/discrete_losses.npy', np.array(discrete_losses))
    np.save(f'{results_path}/cont_losses.npy', np.array(cont_losses))
    np.save(f'{results_path}/critic_losses.npy', np.array(critic_losses))
    np.save(f'{results_path}/entropy_coeff.npy', np.array(entropy_coeff))
    np.save(f'{results_path}/action_std.npy', np.array(action_std))

    agent.save_models(results_path)


# TEST POLICY
def test(env, agent, deterministic=True):
    state, _ = env.reset()
    state = np.array(reduce_state(state), dtype=np.float32, copy=False)
    episode_reward = 0
    step = 0

    for _ in range(100):
        # get next actions
        if deterministic:
            discrete_action = agent.get_discrete_det_action(state)
            cont_action = agent.get_continuous_det_action(state) # (1, 3)
        else:
            discrete_action, discrete_log_prob = agent.get_discrete_action(state)
            cont_action, cont_log_prob = agent.get_continuous_action(state) # (1, 3)
        
        action = convert_action(discrete_action.item(), np.array(cont_action, dtype=np.float32, copy=False)[0])
        
        # agent takes action
        (next_state, steps), reward, terminal, _ = env.step(action)
        step += 1
        episode_reward += reward

        # go to next state
        state = np.array(reduce_state(next_state), dtype=np.float32, copy=False)

        ##env.render()
        #time.sleep(0.02)
        
        if terminal:
            break
    
    episode_reward = round(episode_reward, 4)
    print(f'steps: {step} episode reward: {episode_reward}')
    return episode_reward

    
def main():
    # create environment
    env = gym.make('Platform-v0')
   

    if random_seed:
        print("random seed: ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    #print("observation space: ", env.observation_space)
    #print("action space: ", env.action_space)

    # Create agent
    agent = HPPO(num_states, env.action_space, action_std_init, seed=random_seed)
    
    if only_test_model:
        agent.load_models(results_path)
    else:
        train(env, agent)

    evals = []
    for _ in range(100):
        r = test(env, agent, deterministic=True)
        evals.append(r)
    print(f"avg: {np.mean(evals)} max: {max(evals)} min: {min(evals)}")
    
    # stochastic policy test
    #agent.action_std = 0.1
    #print("stochastic policy test")
    
    #evals = []
    #for _ in range(10):
    #    r = test(env, agent, deterministic=False)
    #    evals.append(r)
    #print(f"avg: {np.mean(evals)} max: {max(evals)} min: {min(evals)}")


    env.close()
        

# Main entry point
if __name__ == "__main__":
    main()
    