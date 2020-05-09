import argparse
import sys
from PIL import Image, ImageOps
import gym
from gym import wrappers, logger
from Debugger import Debugger
import pybulletgym
import numpy as np

e,d = exec, Debugger()
e(d.gIS())


def showImageArray(image):
    im = Image.fromarray(image)
    im.show()
def preprocess(image, scale = 0.525, grayScale = True):
    imPil = Image.fromarray(image)
    if grayScale: imPil = ImageOps.grayscale(imPil)
    if scale != 1: imPil  = imPil.resize((int(imPil.width * scale),int( imPil.height * scale)), Image.BICUBIC)
    
    imArray = np.asarray(imPil)
    
    showImageArray(imArray)
    imArray = imArray[26:110,:]
    showImageArray(imArray)
    return imArray
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
    ob = preprocess(ob)
    e(g('type(ob)'))
    e(g('ob.shape'))	
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
                
        if done: break
env.close()
	
