import torch
import numpy as np
from actor_critic import ActorCritic
import torch.nn as nn
from grid import Grid
import logging
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)

class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def reset(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []


# solutions from this ppo implementation are taken and apapted from https://github.com/nikhilbarhate99/PPO-PyTorch
class PPO:
    def __init__(self, state_space, action_space, n_latent_var, layers, lr, betas, update_iter, clip):

        self.lr = lr
        self.betas = betas
        self.clip = clip
        self.update_iter = update_iter

        self.state_space = state_space
        self.action_space = action_space

        self.policy = ActorCritic(state_space, action_space, n_latent_var, layers).to(device)
        self.policy_old = ActorCritic(state_space, action_space, n_latent_var, layers).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.val_loss_fun = nn.MSELoss()

        self.policy_old.load_state_dict(self.policy.state_dict())

    """
    @eb entropy bonus
    """

    def update(self, buffer, eb=0.01):
        # message m-1 -> loc -> r, two step reward
        rewards = buffer.rewards
        for i in range(len(rewards) - 1):
            rewards[i] += rewards[i + 1]

        # normalizing rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(buffer.states).to(device).detach()
        old_actions = torch.stack(buffer.actions).to(device).detach()
        old_logprobs = torch.stack(buffer.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for k in range(self.update_iter):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            # reshape advantages such that there is a value for each ratio/column
            advantages = advantages.repeat(self.action_space[0], 1).T

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            val_loss = self.val_loss_fun(state_values, rewards)
            loss = -torch.min(surr1, surr2) + 0.5 * val_loss - eb * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # log loss and entropy
            logging.info("iter " + str(k) + " policy loss " + str(loss.mean()) + " value loss " + str(val_loss))

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main(epochs=100, epoch_steps=8000, population=2, lr=0.001, world_size=3, patches=3, n_latent_var=128, layers=1,
         msg_len=3, random=False):
    env = Grid(world_size=world_size, patches=patches)

    talker_state_space = (3, 1)  # location + quantity
    talker_action_space = (msg_len, world_size)  # msg length, vocabulary
    mover_state_space = (msg_len*population,1)  # messages of all agents
    mover_action_space = (2, world_size)  # location in coordinates


    betas = (0.9, 0.999)
    update_iter = 4  # update policy for n epochs
    clip = 0.2  # clip parameter for PPO

    talker_buffer = Buffer()
    talker_policy = PPO(talker_state_space,talker_action_space, n_latent_var, layers, lr, betas, update_iter, clip)

    mover_buffer = Buffer()
    mover_policy = PPO(mover_state_space, mover_action_space, n_latent_var, layers, lr, betas, update_iter, clip)

    logging.info("Population: " + str(population) + " Grid: " + str(world_size) + " Patches: " + str(patches) +
                 " Layers" + str(layers) + " Layer Size: " + str(n_latent_var) + " Msg length: " + str(msg_len))

    # logging variables
    reward_history = []
    running_reward = 0

    for epoch in range(epochs):
        # setup
        env.setup_grid()
        messages = []
        for t in range(epoch_steps):
            step_reward = 0
            new_messages = []
            done = False
            for agent in range(population):
                # state is different for each agent since observation are different
                if len(messages) == 0:
                    messages = np.random.randint(0, world_size, mover_state_space[0])

                #pi(location|messages)
                loc = mover_policy.policy_old.act(np.array(messages), mover_buffer)
                loc = [loc.detach().numpy()] #in list cause grid expects list of locations

                #agent-env interaction
                obs, rew, done = env.step(loc, random=random)

                #shared communication reward for all agents
                step_reward += rew

                #individual exploration reward for all agents
                mover_buffer.rewards.append(rew)

                #pi(msg|obs)
                msg = talker_policy.policy_old.act(np.array(obs), talker_buffer)
                msg = msg.detach().numpy()

                #broadcast message to population
                new_messages.append(msg)

            # reset grid if all patches are foraged
            if done:
                env.setup_grid()
                new_messages = []

            #messages for next step
            messages = np.ravel(new_messages)

            #add reward for each agent trajectory:
            for agent in range(population):
                talker_buffer.rewards.append(step_reward)

            running_reward += step_reward

        talker_policy.update(talker_buffer)
        mover_policy.update(mover_buffer)
        talker_buffer.reset()
        mover_buffer.reset()

        # logging
        running_reward = (running_reward / epoch_steps)
        logging.info("Epoch: " + str(epoch) + " Avg reward: " + str(running_reward))
        if running_reward < 0.3 and epoch >= 200:
            logging.info("cutoff training")
            break
        reward_history.append(running_reward)
        running_reward = 0



    return reward_history


if __name__ == '__main__':
    results = []

    # grid with high/low chance of success
    # small medium big population
    # 0-3-10 messages
    results.append(main(epochs=500, epoch_steps=4000, population=2, lr=0.001, world_size=3, patches=3, n_latent_var=8, layers=2,
         msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=2, lr=0.001, world_size=3, patches=1, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=2, lr=0.001, world_size=6, patches=2, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=5, lr=0.001, world_size=3, patches=3, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=5, lr=0.001, world_size=3, patches=1, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=5, lr=0.001, world_size=6, patches=2, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=10, lr=0.001, world_size=3, patches=3, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=10, lr=0.001, world_size=3, patches=1, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=10, lr=0.001, world_size=6, patches=2, n_latent_var=8, layers=2,
             msg_len=3))
    results.append(
        main(epochs=500, epoch_steps=4000, population=2, lr=0.001, world_size=3, patches=3, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=2, lr=0.001, world_size=3, patches=1, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=2, lr=0.001, world_size=6, patches=2, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=5, lr=0.001, world_size=3, patches=3, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=5, lr=0.001, world_size=3, patches=1, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=5, lr=0.001, world_size=6, patches=2, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=10, lr=0.001, world_size=3, patches=3, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=10, lr=0.001, world_size=3, patches=1, n_latent_var=8, layers=2,
             msg_len=5))
    results.append(
        main(epochs=500, epoch_steps=4000, population=10, lr=0.001, world_size=6, patches=2, n_latent_var=8, layers=2,
             msg_len=5))
    print(results)





