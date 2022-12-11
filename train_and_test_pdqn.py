import time
import numpy as np
import torch
import gym
import gym_platform
import matplotlib.pyplot as plt
import os
from agents.pdqn import PDQNAgent


# folder to save results
test_num = 1
results_path = f"pdqn_results/{test_num}"
os.makedirs(results_path, exist_ok=True)

random_seed = None
only_test_model = False
num_states = 4

PARAMETERS_MIN = np.array([0, 0, 0])
PARAMETERS_MAX = np.array([
    30,  # run
    720,  # hop
    430  # leap
])

# HYPERPARAMETERS
max_episodes = 10000
update_freq = 50

explore_step_freq = 20
explore_episode_freq = 30


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

    max_episode_len = 30
    num_episodes = 0
    total_reward = 0
    returns = []
    returns_avg_20 = []
    eval_returns = []
    avg_param_actor_losses = []
    avg_q_network_losses = []

    save_freq = 0
    step = 0

    r_avg_100 = 0
    eval_steps_avg = 0
    explore_step_freq = 20
   

    for _ in range(max_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        state = reduce_state(state)

        episode_reward = 0
        episode_steps = 0
        exploring = False

        if num_episodes % explore_episode_freq == 0:
            exploring = True

        
        agent.start_episode()
       
        
        for _ in range(max_episode_len):
            #if (step % explore_step_freq) == 0 or (exploring and episode_reward > r_avg_100):
            #    act, act_param, all_action_parameters = agent.get_next_action(state, epsilon=1)
            #    print("exploring")
            #else:
            act, act_param, all_action_parameters = agent.get_next_action(state)
            
            action = convert_action(act, act_param) # scale to environment (action_index, (dxa0, dxa1, dxa2))

            (next_state, steps), reward, terminal, _ = env.step(action)
            next_state = np.array(reduce_state(next_state), dtype=np.float32, copy=False)
            
            # Get next action from policy
            next_act, next_act_param, next_all_action_parameters = agent.get_next_action(next_state)
            next_action = convert_action(next_act, next_act_param)

            # add transition to agent replay buffer and optimise networks
            agent.step(state, (act, all_action_parameters), reward, next_state, terminal)

            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            step += 1
            episode_steps += 1
            episode_reward += reward

            if terminal:
                break
        
        agent.end_episode()
        returns.append(episode_reward)
        total_reward += episode_reward
        num_episodes += 1
        

        if num_episodes % 20 == 0:
            returns_avg_20.append(np.mean(returns[-20:]))
            avg_param_actor_losses.append(np.mean(agent.param_actor_losses))
            avg_q_network_losses.append(np.mean(agent.q_network_losses))
            # clear
            agent.param_actor_losses = []
            agent.q_network_losses = []

        if num_episodes % 100 == 0:
            r_avg_100 = np.percentile(returns[-100:], 60)
            print(f'episode {num_episodes} episode reward: {episode_reward} r100: {np.array(returns[-100:]).mean()} epsilon: {agent.epsilon}')
        
        if num_episodes % 500 == 0:
            evals = []
            for _ in range(10):
                r, n_steps = test(env, agent, deterministic=True)
                evals.append(r)
                eval_steps_avg += n_steps
            eval_returns.append([np.mean(evals), max(evals), min(evals)])
            eval_steps_avg //= 10

            #explore_step_freq = min(explore_step_freq + 10, 100)

        agent.update_epsilon()


    # save results
    np.save(f'{results_path}/returns.npy', np.array(returns_avg_20))
    np.save(f'{results_path}/eval_returns.npy', np.array(eval_returns))
    np.save(f'{results_path}/param_actor_losses.npy', np.array(avg_param_actor_losses))
    np.save(f'{results_path}/q_network_losses.npy', np.array(avg_q_network_losses))
    

def test(env, agent, deterministic=True):
    # TEST POLICY
    state, _ = env.reset()
    state = np.array(reduce_state(state), dtype=np.float32, copy=False)
    episode_reward = 0
    step = 0

    act, act_param, all_action_parameters = agent.get_greedy_action(state)
    action = convert_action(act, act_param)

    for _ in range(100):
        
        (next_state, steps), reward, terminal, _ = env.step(action)
        next_state = np.array(reduce_state(next_state), dtype=np.float32, copy=False)
        
        # get action
        if deterministic:
            next_act, next_act_param, next_all_action_parameters = agent.get_greedy_action(next_state)
        else:
            next_act, next_act_param, next_all_action_parameters = agent.get_next_action(next_state)
        next_action = convert_action(next_act, next_act_param)

        agent.step(state, (act, all_action_parameters), reward, next_state, terminal)
        act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
        action = next_action
        state = next_state

        step += 1
        episode_reward += reward

        #env.render()
        #time.sleep(0.002)
        
        if terminal:
            break
    
    print(f'steps: {step} episode reward: {episode_reward}')
    return episode_reward, step
    

def main():
    # create environment
    env = gym.make('Platform-v0')
    #print("observation space: ", env.observation_space)
    #print("action space: ", env.action_space)

    if random_seed:
        print("random seed: ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
        if device == torch.device("cuda"):
                torch.cuda.manual_seed(random_seed)

    # Create agent
    agent = PDQNAgent(num_states, env.action_space, seed=random_seed)
    
    if only_test_model:
        agent.load_models(results_path)
    else:
        train(env, agent)
    
    test(env, agent)

    env.close()

        

# Main entry point
if __name__ == "__main__":
    main()
    