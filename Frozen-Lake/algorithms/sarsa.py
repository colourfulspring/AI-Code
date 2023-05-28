import numpy as np
import gymnasium as gym


class Sarsa(object):
    def __init__(self, lr, epsilon, gamma, state_size, action_size):
        self.tabular = np.random.normal(0, 0.1, (state_size, action_size))
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma

    def epsilon_greedy_policy(self, s):
        num_actions = self.tabular.shape[1]
        random_num = np.random.random()
        if random_num < self.epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(self.tabular[s])
        return action

    def greedy_policy(self, s):
        action = np.argmax(self.tabular[s])
        return action

    def learn_from_tuple(self, s, a, r, next_s, next_a):
        self.tabular[s][a] = self.tabular[s][a] + self.lr * (
                r + self.gamma * self.tabular[next_s][next_a] - self.tabular[s][a])


def runSarsa(config):
    # initialization
    if config.map == '4x4':
        maze_size = 4
    else:
        maze_size = 8

    td_algo = Sarsa(config.lr, config.epsilon, config.gamma, maze_size ** 2, 4)

    # training
    env = gym.make('FrozenLake-v1', is_slippery=False)
    rewards = []
    avg_rewards = []
    for ep_i in range(0, config.episodes):
        obs, info = env.reset(seed=config.seed)
        action = td_algo.epsilon_greedy_policy(obs)  # this is where you would insert your policy
        tot_reward = 0
        pow_gamma_n = 1.0
        while True:
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_action = td_algo.epsilon_greedy_policy(obs)  # this is where you would insert your policy
            td_algo.learn_from_tuple(obs, action, reward, next_obs, next_action)
            tot_reward += pow_gamma_n * reward
            pow_gamma_n *= config.gamma

            if terminated or truncated:
                break
            obs = next_obs
            action = next_action
        print(f'Episodes {ep_i}: Rewards {tot_reward}')
        rewards.append(tot_reward)
        if ep_i % 100 == 0:
            avg_rewards.append(sum(rewards) / len(rewards))
    env.close()

    # testing
    for i in range(0):
        env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)
        obs, info = env.reset(seed=config.seed)
        while True:
            print(f'obs={obs}')
            action = td_algo.greedy_policy(obs)  # this is where you would insert your policy
            print(f'action={action}')
            obs, reward, terminated, truncated, info = env.step(action)
            print(f'reward={reward}')
            env.render()

            if terminated or truncated:
                break
        env.close()
    return avg_rewards
