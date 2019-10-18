import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, env,gamma=0.9):
        super(Policy, self).__init__()
        self.state_space = env.state_space[0]
        self.action_space = env.action_space

        #actor + critic layer
        self.head = nn.Linear(self.state_space,4)

        self.actionl = nn.Linear(4,env.action_space[0] * env.action_space[1])
        self.criticl = nn.Linear(4,1)

        self.gamma = gamma

        # episode policy and reward history
        self.policy_history = []
        self.reward_episode = []

    def forward(self, x):
        print(x.shape, self.head)
        x = F.relu(self.head(x))
        #actor
        a_probs = F.softmax(self.actionl(x).view(self.action_space),dim=-1)
        #critic
        val = self.criticl(x)
        return a_probs, val

#this piece of code is inspiried from https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
def update_policy(policy, optimizer):
    #immedeate step reward
    rewards = policy.reward_episode

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() +
                                            np.finfo(np.float32).eps)

    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    for (logp_a, val), R in zip(policy.policy_history, rewards):
        advantage = R - val.item()

        #actor loss
        policy_losses.append(-logp_a * advantage)

        #critic losses
        value_losses.append(F.smooth_l1_loss(val, torch.Tensor([R])))

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # Update network weights
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()



def select_action(policy, state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    a_probs, val  = policy(state)
    a_probs = a_probs.reshape(policy.action_space)
    c = Categorical(a_probs)
    try:
        action = c.sample()
        #if action ==1:
            #print(action,out)
    except:
        print("ERROR")
        print("reshaped out",a_probs)
        print("state", state)
        print("output",policy(state))
        print("c", c)
        print("params",policy.named_parameters())

    # add log prob of the chosen action and critic value to the history
    policy.policy_history.append((c.log_prob(action),val))
    return action
