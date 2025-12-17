
import torch
import numpy as np



def game_step(observation, env, agent, device, train=True):

    
    # agent takes action based on observation 
    action = agent.act(observation)

    # after taking the action, observing the next observation and reward
    observation_, reward, done, _, info = env.step(action)
    reward = torch.tensor([reward], device=device)
    done_tensor = torch.tensor([done], dtype = torch.float32, device = device)
    agent.memory.push(observation, torch.tensor([action], device=device), reward, observation_, done_tensor)
    
    if train:
        agent.optimize()

    return observation_, done
    
    

def run(sharpe_matrix, epoch, agent, env, device, episodes, t_upfreq):
    max_profit = -99999999
    step = 0

    for episode in range(episodes):
        # initial observation
        observation, info = env.reset()
        done = False
        
        while not done:
            observation, done = game_step(observation, env, agent, device)
            step += 1
                
        if env.total_profit > max_profit:
            torch.save(agent.policy_net.state_dict(), 'models/DQN_best_model')
            max_profit = env.total_profit
        
        # Calculate Sharpe Ratio
        daily_ret = env.daily_return_account
        sharpe = np.mean(daily_ret) / np.std(daily_ret) * (252 ** 0.5)
        sharpe_matrix[epoch][episode] = sharpe # for plotting performance graphs over episodes
        
        print('episode:%d, total_profit:%.3f, sharpe:%.3f' % (episode, env.total_profit, sharpe))

         # Check if it's time to update the target network
        if episode > 0 and episode % t_upfreq == 0:
            # Update the target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    return sharpe_matrix

def BackTest(agent, env, device):
    # Load best model
    agent.policy_net.load_state_dict(torch.load('models/DQN_best_model'))
    # Run validation
    observation, info = env.reset()
    while True: 
        observation, done = game_step(observation, env, agent, device, train=False)
        if done:
            break
    return env