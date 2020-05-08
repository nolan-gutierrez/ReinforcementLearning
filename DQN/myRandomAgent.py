import argparse
import sys
import gym
from gym import wrappers, logger
from Debugger import Debugger
import pybulletgym
e,d = exec, Debugger()
e(d.gIS())


class RandomAgent(object):
	"""
	The world's simplest agent!
	"""
	def __init__(self, action_space):
		self.action_space = action_space
	def act(self, observation, reward, done):
		return self.action_space.sample()
parser = argparse.ArgumentParser(description = None)
parser.add_argument('env_id', nargs= '?', default = 'Pong-v0', help='Select the environment to run')

args = parser.parse_args()
logger.set_level(logger.INFO)
env = gym.make(args.env_id)
outdir = '/tmp/random-agent-results'
env = wrappers.Monitor(env,directory = outdir, force = True)
env.seed(0)
agent = RandomAgent(env.action_space)
episode_count = 1000
reward = 0
done = False
for i in range(episode_count):
	env.render()
	ob = env.reset()
	e(g('type(ob)'))
	e(g('ob.shape'))	
	e(g('ob'))
	while True:
		action = agent.act(ob, reward, done)
		ob, reward, done, _ = env.step(action)
		if done: break
env.close()
	
