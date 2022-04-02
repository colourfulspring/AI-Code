import gym
import torch
from torch import nn


def normalize_state(state):
    condition = torch.tensor([True, False, True, False], device=device)
    length = torch.tensor([4.8, 1.0, 0.418, 1.0], device=device)
    return torch.where(condition, state / length, torch.tanh(state))


gamma = 0.9
episodes = 10000
env = gym.make('CartPole-v1')
device = 'cuda' if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.preNet = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
        )
        self.actorNet = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax(dim=0)
        )
        self.criticNet = nn.Linear(128, 1)

    def forward(self, x):
        return self.actorNet(self.preNet(x)), self.criticNet(self.preNet(x))


net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for episode in range(episodes):
    state = torch.tensor(env.reset(), device=device)
    state = normalize_state(state)
    gammas = 1
    reward_sum = 0
    while True:
        env.render()
        policy, value = net(state)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()  # action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor(next_state, device=device)
        next_state = normalize_state(next_state)
        with torch.no_grad():
            _, next_value = net(next_state)
            if done:
                next_value = torch.tensor(0.0, device=device)
            delta = reward + gamma * next_value - value

        optimizer.zero_grad()
        loss_func = nn.HuberLoss()
        value_loss = loss_func(reward + gamma * next_value, value)
        policy_loss = -gammas * delta * dist.log_prob(action)
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()

        reward_sum += reward
        if done:
            print("episode: " + str(episode) + "    reward:" + str(reward_sum))
            break

        gammas = gammas * gamma
        state = next_state

env.close()
