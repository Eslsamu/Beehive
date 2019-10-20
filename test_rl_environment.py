from ppo import PPO, Buffer
import numpy as np

class TestEnv:
    def __init__(self):
        self.state_space = (1, 1)
        self.action_space = (1, 10)
        self.num = 0

    def step(self, action):
        if action == self.num:
            rew = 1.
        else:
            rew = 0

        self.num = np.random.randint(0, 10)
        obs = self.num

        return obs, rew

env = TestEnv()
ppo = PPO()
buffer = Buffer()

for epoch in range(50):
    state = env.step(5)[0]
    epoch_reward = 0
    for t in range(1000):
        # Running policy_old:
        action = ppo.policy_old.act(np.array([state]), buffer)
        state, reward = env.step(action)

        # Saving reward and is_terminal:
        buffer.rewards.append(reward)
        epoch_reward += reward

    ppo.update(buffer)
    buffer.reset()
    print(epoch_reward)
