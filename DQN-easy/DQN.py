# -*- coding: utf-8 -*-
# @Software: PyCharm
# @Time    : 2023/12/21 20:16
# @Author  : Jane_Sang
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import datetime
import numpy as np
import gym
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from envx import TaskOffloadingEnv

writer = SummaryWriter('./log/{}'.format(datetime.datetime.now().__str__().replace(':', '_'))+" DQNall")

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    total_reward = 0
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        taken_actions = set()
        for step in range(max_steps):
            action = agent.get_action(state,taken_actions,episode)
            taken_actions.add(action)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward


            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state
        total_reward += episode_reward
        writer.add_scalar('episode/reward', episode_reward, episode + 1)
        writer.add_scalar('episode/avg-reward', total_reward / (episode + 1), episode + 1)

    return episode_rewards


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class DQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=0.001, gamma=0.99, tau=0.01, buffer_size=100000,epsilon_start=0.5, epsilon_end=0.01, epsilon_decay=500):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.target_model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model = DQN(state_dim, action_dim).to(self.device)
            self.target_model = DQN(state_dim, action_dim).to(self.device)

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * episode / self.epsilon_decay)
    def get_action(self, state,  taken_actions,episode):
        self.update_epsilon(episode)
        if (np.random.randn() < self.epsilon):
            action = self.env.action_space.sample()
            while action in taken_actions:
                action = self.env.action_space.sample()
            return action

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        qvals_np = qvals.cpu().detach().numpy()

        available_actions = [i for i in range(len(qvals_np[0])) if i not in taken_actions]
        action = available_actions[np.argmax(qvals_np[0][available_actions])]

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        loss = F.mse_loss(curr_Q, expected_Q.detach())

        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)



MAX_EPISODES = 10000
MAX_STEPS = 1000
BATCH_SIZE = 64

env = TaskOffloadingEnv(use_en=True, use_bs=True)
n_en = env.n_en
n_ue = env.n_ue
state_dim = 46


action_dim = n_en + n_ue
agent = DQNAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)