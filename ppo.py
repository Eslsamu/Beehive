import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space, n_latent_var):
        super(ActorCritic, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        # head
        self.head = nn.Linear(self.state_space[0], 4)

        # actor
        self.actionl = nn.Linear(4, action_space[0] * action_space[1])

        # critic
        self.criticl = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.head(x))
        # actor
        a_probs = F.softmax(self.actionl(x).view(-1, self.action_space[1]), dim=1)
        # critic
        val = self.criticl(x)
        return a_probs, val

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.forward(state)[0]
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        # TODO potential error
        action_probs = action_probs.view((-1, self.action_space[0], self.action_space[1]))
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_space, action_space, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_space, action_space, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_space, action_space, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = memory.rewards
        for i in range(len(rewards) - 2):
            # message loc-2 -> m-1 -> loc, two step reward
            rewards[i] += rewards[i + 1] + rewards[i + 2]

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            # reshape advantages such that there is a value for each ratio/column
            advantages = advantages.repeat(5, 1).T
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class Run():
    def __init__(self, population, world_size=10, patches=10, max_food_quant=5, random_quant=False):
        self.msg_len = 3
        self.loc_len = 2
        self.loc_info_len = 1
        self.action_space = (self.msg_len + self.loc_len, world_size)
        self.state_space = (self.loc_len + self.loc_info_len + self.msg_len * population, 1)
        self.world_size = world_size
        self.patches = patches
        self.max_food_quant = max_food_quant
        self.random_quant = random_quant
        self.setup_grid()

    def setup_grid(self):
        self.world = np.zeros((self.world_size, self.world_size))
        # a patch is marked in the world grid by the quantity of food stored
        for p in range(self.patches):
            x = np.random.randint(0, self.world_size)
            y = np.random.randint(0, self.world_size)
            if self.random_quant:
                q = np.random.randint(1, self.max_food_quant)
            else:
                q = self.max_food_quant
            self.world[x][y] = q

    def step(self, locations, random=False):
        # global reward
        rew = 0

        # explored locations
        obs = []

        for l in locations:
            x = l[0]
            y = l[1]

            if random:
                x = np.random.randint(0, self.world_size)
                y = np.random.randint(0, self.world_size)

            # collects food at location that agent picked if there is some
            q = self.world[x][y]
            if q > 0:
                rew += 1
                self.world[x][y] -= 1
            else:
                # TODO adjust punishment
                rew -= 0.1

            # store explored location
            obs.append([x, y, q])

        done = True
        if np.any(self.world): done = False

        return np.array(obs), rew, done


def main(population=5, max_episodes=10000, lr=0.005, max=0, random=False):
    env = Run(population)
    state_space = env.state_space
    action_space = env.action_space
    log_interval = 20
    n_latent_var = 128  # number of variables in hidden layer
    update_timestep = 4000  # update policy every n timesteps
    betas = (0.9, 0.999)
    gamma = 0  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO

    memory = Memory()
    ppo = PPO(state_space, action_space, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    # TODO agents
    # TODO test and write report of all results till now
    for i_episode in range(1, max_episodes + 1):

        # setup
        env.setup_grid()
        old_messages = []
        obs = None

        for t in range(1000):
            timestep += 1

            locations = []
            messages = []
            for agent in range(population):

                # state is different for each agent since observation are different
                if obs is None:
                    state = np.zeros(state_space[0])
                else:
                    state = np.concatenate((obs[agent], (np.array(old_messages).ravel())))

                # Running policy_old:
                a = ppo.policy_old.act(state, memory)
                a = a.detach().numpy()

                # seperate location and message
                loc = a[:2]
                msg = a[2:]
                locations.append(loc)
                messages.append(msg)

            # store messages to use as state in next step
            old_messages = messages

            obs, rew, done = env.step(locations)

            # Saving reward and is_terminal:
            for agent in range(population):
                memory.rewards.append(rew)

            # update if its time
            if timestep % update_timestep == 0:
                print("update")
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            if done:
                rew += 20
                break

            running_reward += rew

        avg_length += t

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()


def maintest(max_episodes=10000, lr=0.01, max=0, random=False):
    env = TestEnv()
    state_dim = 1
    action_dim = 10
    log_interval = 1000  # print avg reward in the interval
    n_latent_var = 128  # number of variables in hidden layer
    update_timestep = 5000  # update policy every n timesteps
    lr = lr
    betas = (0.9, 0.999)
    gamma = 0  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.step(5)[0]
        for t in range(1000):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(np.array([state]), memory)
            state, reward = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward

        avg_length += t

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


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