import numpy as np
from policy import Policy, update_policy, select_action
import torch
import logging
import sys

def simple_test_policy(epochs, epoch_steps, lr=0.01):

    class TestEnv:
        def __init__(self):
            self.state_space = (1,1)
            self.action_space = (1,10)
            self.num = 0

        def step(self,action):
            if self.num == action:
                rew = 1
            else:
                rew = -1

            self.num = np.random.randint(0,10)
            obs = self.num

            return  obs, rew

    logging.basicConfig(stream=sys.stdout, filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    env = TestEnv()
    policy = Policy(env)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_rew = 0
        obs = env.step(5)[0]
        for step in range(epoch_steps):
            action = select_action(policy, np.array([obs]))
            obs, rew = env.step(action)
            logging.log(logging.DEBUG, "obs " + str(obs) + " rew " + str(rew))
            epoch_rew += rew
            policy.reward_episode.append(rew)
        update_policy(policy, optimizer)
        logging.log(logging.INFO, "epoch " + str(epoch) + " total reward " + str(epoch_rew))


    for i in range(10):
        env.num = i
        obs = i
        action = select_action(policy,np.array([obs]))
        _, rew = env.step(action)
        print(i, rew)

class Run():
    def __init__(self, population, world_size=4, patches=8, max_food_quant=5, random_quant = False):
        self.msg_len = 3
        self.loc_len = 2
        self.loc_info_len = 1
        self.action_space = (self.msg_len+self.loc_len, world_size)
        self.state_space = (self.loc_len + self.loc_info_len + self.msg_len*population,1)
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


    def step(self, locations):
        # global reward
        rew = 0

        # explored locations
        obs = []

        for l in locations:
            x = l[0]
            y = l[1]
            # collects food at location that agent picked if there is some
            q = self.world[x][y]
            if q > 0:
                rew += 1
                self.world[x][y] -= 1
            else:
                rew -= 0.1

            # store explored location
            obs.append([x, y, q])

        done = True
        if np.any(self.world): done = False

        return np.array(obs), rew, done

#TODO track reward per episode
def train(epochs, epoch_steps, population=3, lr = 0.01,debug = False):
    run = Run(population)
    policy = Policy(run)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    #logging for debugging and learning info
    if debug:
        logging.basicConfig(filename=train.log, filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)

    logging.log(logging.INFO,"################logging############")
    logging.log(logging.INFO,"policy size "+str(sum(p.numel() for p in policy.parameters()))
                            +"\naction space: "+str(policy.action_space)
                            +"\nstate space: "+str(policy.state_space))

    for epoch in range(epochs):
        #epoch reward
        total_rew = 0

        #setup
        run.setup_grid()
        old_messages = []
        obs = None

        logging.log(logging.INFO, "Epoch "+str(epoch))

        for step in range(epoch_steps):
            logging.log(logging.DEBUG, "====== step "+ str(step)+ "========")

            locations = []
            messages = []
            for agent in range(population):

                logging.log(logging.DEBUG,"agent "+str(agent))

                #state is different for each agent since observation are different
                if obs is None:
                    state = np.zeros(policy.state_space)
                else:
                    print(obs[agent], old_messages)
                    state = np.concatenate((obs[agent],(np.array(old_messages).ravel())))
                    logging.log(logging.DEBUG,"agent state"+ str(state))

                a = select_action(policy, state)
                a = a.detach().numpy()

                #seperate location and message
                loc = a[:2]
                msg = a[2:]

                logging.log(logging.DEBUG,"loc "+ str(loc))
                logging.log(logging.DEBUG,"message "+str(msg))

                locations.append(loc)
                messages.append(msg)

            #store messages to use as state in next step
            old_messages = messages

            obs, rew, done = run.step(locations)

            #episode info
            total_rew += rew

            # Save reward for each agent
            for agent in range(population):
                policy.reward_episode.append(rew)
            if done:
                run.setup_grid()
                old_messages = []
                obs = None

            logging.log(logging.DEBUG,"obs " +str(obs)+ " rew "+ str(rew)+" done "+str(done))
            logging.log(logging.DEBUG,str(run.world))

        logging.log(logging.INFO,"epoch " +str(epoch)+" total reward "+str(total_rew))

        update_policy(policy, optimizer)



def ran_run(runs,steps, population=3):
    run = Run(population)
    run.setup_grid()
    total_rew = 0
    for r in range(runs):
        run_reward = 0
        for step in range(steps):
            locations = np.random.randint(0, run.world_size, (population,2))
            _, rew, done = run.step(locations)
            run_reward += rew
            if done:
                run.setup_grid()
        total_rew += run_reward

    return total_rew/runs

#simple_test_policy(100,10000, lr = 0.03)
#print(ran_run(100,4000))
train(100, 4000)
