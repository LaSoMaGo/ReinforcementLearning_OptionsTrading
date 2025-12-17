import torch
import numpy as np



def game_step(observation, env, agent, device, train=True):
    # Agent takes action based on observation 
    action, log_prob, entropy = agent.act(observation)

    # Take action and observe next state and reward
    observation_, reward, done, _, info = env.step(action)
    reward = torch.tensor([reward], device=device)
    done_tensor = torch.tensor([done], dtype=torch.float32, device=device)

    agent.memory.push(observation, torch.tensor([action], device=device), reward, observation_, done_tensor)

    # Store transition in memory
    if train:
        agent.optimize()

    return observation_, done 

def run(sharpe_matrix, epoch, agent, env, device, episodes, t_upfreq):
    max_profit = -99999999
    step = 0

    for episode in range(episodes):
        observation, info = env.reset()
        done = False

        while not done:
            observation, done = game_step(observation, env, agent, device)
            step += 1

        if env.total_profit > max_profit:
            torch.save(agent.actor.state_dict(), 'models/A2C_best_actor_model')
            torch.save(agent.critic.state_dict(), 'models/A2C_best_critic_model')
            max_profit = env.total_profit
     

         # Calculate Sharpe ratio
        daily_ret = env.daily_return_account
        if np.std(daily_ret) != 0:
            sharpe = np.mean(daily_ret) / np.std(daily_ret) * (252 ** 0.5)
        else:
            sharpe = 0
        sharpe_matrix[epoch][episode] = sharpe

        print('episode:%d, total_profit:%.3f, sharpe:%.3f' % (episode, env.total_profit, sharpe))

            
    return sharpe_matrix


def BackTest(agent, env):
    # Load best model
    agent.actor.load_state_dict(torch.load('models/A2C_best_actor_model'))
    agent.critic.load_state_dict(torch.load('models/A2C_best_critic_model'))
    # Run validation
    observation, info = env.reset()
    while True:
        observation, done = game_step(observation, env, agent, train=False)
        if done:
            break
    return env
