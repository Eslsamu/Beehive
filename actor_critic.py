import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space, n_latent_var, layers):
        super(ActorCritic, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        # head
        self.head = nn.Linear(self.state_space[0], n_latent_var)

        # hidden
        self.hiddenl = nn.ModuleList([nn.Linear(n_latent_var,n_latent_var) for i in range(layers-1)])

        #actor
        self.actionl = nn.Linear(n_latent_var, action_space[0] * action_space[1])

        # critic
        self.criticl = nn.Linear(n_latent_var, 1)

    def forward(self, x):
        x = F.relu(self.head(x))

        for l in self.hiddenl:
            x = F.relu(l(x))

        # actor
        a_probs = F.softmax(self.actionl(x).view(-1, self.action_space[1]), dim=1)
        # critic
        val = self.criticl(x)
        return a_probs, val

    def act(self, state, buffer):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.forward(state)[0]
        dist = Categorical(action_probs)
        action = dist.sample()
        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.logprobs.append(dist.log_prob(action))
        return action

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        # TODO potential error
        action_probs = action_probs.view((-1, self.action_space[0], self.action_space[1]))
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy
