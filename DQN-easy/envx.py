import numpy as np
import gym
from gym import spaces

class TaskOffloadingEnv(gym.Env):

    def __init__(self,
                 n_ue=5,
                 n_en=5,
                 n_task=5,
                 use_en=True,
                 use_bs=True,
                 map_size=200):
        super(TaskOffloadingEnv, self).__init__()

        self.use_en = use_en
        self.use_bs = use_bs

        self.n_ue = self.K = n_ue
        self.n_en = self.H = n_en
        self.n_task_per_ue = self.M = n_task
        self.now = 0

        self.map_size = map_size

        # 定义固定的任务属性
        self.tasks_prop = np.array([
            [50, 50, 20, 1, 100],  # [x, y, 截至时间, 优先级, 基础报酬]
            [75, 75, 20, 3, 150],
            [100, 50, 20, 5, 250],
            [125, 35, 20, 7, 200],
            [180, 90, 20, 9, 150]
        ])

        # 定义固定的工人属性
        self.position_ue = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.pay_ue = np.array([
            [11],
            [12],
            [13],
            [14],
            [15]
        ])
        self.ue_prop = np.hstack((self.position_ue, self.pay_ue))

        # 定义固定的无人机属性
        self.position_en = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.pay_en = np.array([
            [20],
            [20],
            [20],
            [20],
            [20]
        ])
        self.en_prop = np.hstack((self.position_en, self.pay_en))

        self.cur_task = 0

        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)

        self.observation_space = spaces.Box(low=0, high=self.map_size, shape=(self.n_task_per_ue * 5 + self.n_ue * 3 + self.n_en * 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_ue + self.n_en)

    def reset(self):
        self.en_prop = np.hstack((self.position_en, self.pay_en))

        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)

        self.cur_task = 0
        self.now = 0

        return self._state()

    def step(self, action):
        reward = self._reward(action)
        reward = np.clip(reward, -1000, np.inf)

        self._update_state(action)

        state = self._state()
        done = self.now >= len(self.tasks_prop)
        return state, reward, done, {}

    def _state(self):
        return np.concatenate((
            self.tasks_prop[self.cur_task],
            [self.cur_task],
            self.resource_ue,
            self.resource_en,
            self.position_ue.flatten(),
            self.position_en.flatten(),
            self.pay_ue.flatten(),
            self.pay_en.flatten()
        ))

    def _reward_ue(self, action):
        task_prop = self.tasks_prop[self.cur_task]
        ue_prop = self.ue_prop[action]
        d0 = np.linalg.norm(self.position_ue[action] - self.tasks_prop[self.cur_task][:2])
        t_ij = d0/3
        get = task_prop[3] * 10 + task_prop[4] + (task_prop[2]-t_ij)*10
        pay = ue_prop[2]
        reward = get - pay
        # 增加基于任务完成进度的奖励
        return reward

    def _reward_en(self, action):
        task_prop = self.tasks_prop[self.cur_task]
        en_prop = self.en_prop[action - 5]
        d0 = np.linalg.norm(self.position_en[action - 5] - self.tasks_prop[self.cur_task][:2])
        speed = 12 #无人机速度
        t_ij = d0 / speed
        get = task_prop[3]*10+ task_prop[4]*1.25+(task_prop[2]-t_ij)*10
        pay = en_prop[2]*t_ij
        reward = get - pay
        # 增加基于任务完成进度的奖励
        return reward

    def _reward(self, action):
        if action < self.n_ue:
            return self._reward_ue(action)
        else:
            return self._reward_en(action)

    def _update_state(self, action):
        if action < self.n_ue:
            self.resource_ue[action] = 0
        else:
            self.resource_en[action - self.n_ue] = 0
        self.cur_task += 1
        if self.cur_task >= len(self.tasks_prop):
            self.cur_task = len(self.tasks_prop) - 1
            self.now = len(self.tasks_prop)


if __name__ == "__main__":
    env = TaskOffloadingEnv()
    state = env.reset()

    for _ in range(5):
        action = np.random.choice(env.action_space.n)
        state, reward, done, _ = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
        if done:
            break
