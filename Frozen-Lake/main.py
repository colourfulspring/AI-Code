import os
import argparse
from algorithms.qlearning import runQlearning
from algorithms.sarsa import runSarsa
from LineChart import LineChart

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def run(config):
    if config.algo == 'sarsa':
        rewards = runSarsa(config)
    else:
        rewards = runQlearning(config)
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gymnasium env name', type=str)
    parser.add_argument('--algo', help='name of algorithm', type=str,
                        choices=['q-learning', 'sarsa'], default='q-learning')
    parser.add_argument('--dir', help='directory to store model', type=str)
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--lr', help='learning rate', type=float, default=5e-2)
    parser.add_argument('--epsilon', help='epsilon-greedy\'s epsilon', default=0.3, type=float)
    parser.add_argument('--episodes', help='total training episodes', default=30000, type=int)
    parser.add_argument('--gamma', help='reward discount factor', default=0.99, type=float)
    parser.add_argument('--map', help='map of frozen lake env', type=str,
                        choices=['4x4', '8x8'], default='4x4')

    # # q-learning, same gamma, different lr
    # gamma = 0.99
    # lr_list = [1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    # linechart = LineChart()
    # for lr in lr_list:
    #     args = ['--lr', f'{lr}', '--gamma', f'{gamma}', '--algo', 'q-learning']
    #     config = parser.parse_args(args)
    #     rewards = run(config)
    #     linechart.add_line(rewards, f'lr={lr}')
    #     print(rewards)
    #
    # linechart.draw_line_chart('episodes*(100)', 'rewards', f'Different lr, gamma={gamma}, Q-learning, FrozenLake-v1')
    # del linechart
    #
    # q-learning, same lr, different gamma
    # lr = 0.5
    # gamma_list = [0.7, 0.8, 0.9, 0.99, 1.0]
    # linechart = LineChart()
    # for gamma in gamma_list:
    #     args = ['--lr', f'{lr}', '--gamma', f'{gamma}', '--algo', 'q-learning']
    #     config = parser.parse_args(args)
    #     rewards = run(config)
    #     linechart.add_line(rewards, f'gamma={gamma}')
    #     print(rewards)
    #
    # linechart.draw_line_chart('episodes*(100)', 'rewards', f'Different gamma, lr={lr}, Q-learning, FrozenLake-v1')
    # del linechart

    # sarsa, same gamma, different lr
    gamma = 0.99
    lr_list = [1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    linechart = LineChart()
    for lr in lr_list:
        args = ['--lr', f'{lr}', '--gamma', f'{gamma}', '--algo', 'sarsa']
        config = parser.parse_args(args)
        rewards = run(config)
        linechart.add_line(rewards, f'lr={lr}')
        print(rewards)

    linechart.draw_line_chart('episodes*(100)', 'rewards', f'Different lr, gamma={gamma}, Sarsa, FrozenLake-v1')
    del linechart

    # # sarsa, same lr, different gamma
    # lr = 0.5
    # gamma_list = [0.7, 0.8, 0.9, 0.99, 1.0]
    # linechart = LineChart()
    # for gamma in gamma_list:
    #     args = ['--lr', f'{lr}', '--gamma', f'{gamma}']
    #     config = parser.parse_args(args)
    #     rewards = run(config)
    #     linechart.add_line(rewards, f'gamma={gamma}')
    #     print(rewards)
    #
    # linechart.draw_line_chart('episodes*(100)', 'rewards', f'Different gamma, lr={lr}, Sarsa, FrozenLake-v1')
    # del linechart
