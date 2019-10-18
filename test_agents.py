from run import Sim
import numpy as np


def test_act(messages, obs):
	msg = obs
	best = 0
	for i,m in enumerate(messages):
		if m[2] > messages[best][2]:
			best = i
	
	if messages[best][2] > 0:
		loc = messages[best][:2]
	else:
		loc = np.random.randint(0,10,2)

	return msg, loc

def act(policy, messages,obs):
	pass

def test():
	sim = Sim()

	n_agents = 5

	messages = np.zeros((n_agents,3))
	obs = np.zeros((n_agents,3))
	locations = np.zeros((n_agents,2),np.int32)

	n_steps = 10
	print("world: ",sim.world)
	for s in range(n_steps):
		new_msgs = np.zeros((n_agents,3))

		for i in range(n_agents):
			msg,loc = test_act(messages, obs[i])
			new_msgs[i] =  msg
			locations[i] = loc

		messages = new_msgs
		obs, rew, done = sim.step(locations)

		print("world:",sim.world, "rew", rew, "obs",obs,"loc", locations,"messages", messages,"done", done)

test()