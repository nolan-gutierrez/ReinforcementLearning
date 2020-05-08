
from PIL import ImageOps,Image
from collections import deque
from Debugger import Debugger
from numpy import random
from Counter import Counter
from DQNModels import DQN
import tensorflow as tf
import argparse
from queue import PriorityQueue
from Plotter import Plot
from collections import defaultdict
import cv2
import numpy as np
from gym import wrappers, logger
import pybulletgym
import gym
e,d = exec, Debugger()
e(d.gIS())
d.sDS(False)
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type = float, default = 0.1)
parser.add_argument('--alpha', type = float, default = 0.1)
parser.add_argument('--gamma',type = float, default = 0.95)
parser.add_argument('--visual',type = bool, default = False)
parser.add_argument('--nEpisodes',type = int, default = 300)
parser.add_argument('--nplanning', type = int, default = 10)
parser.add_argument('--greedy', type = bool, default = False)
parser.add_argument('--epsilonStrat',type = int, default = 1)
parser.add_argument('--learner', type = int, default =1)
parser.add_argument('--showTrial', type = bool, default = True)
parser.add_argument('--randomReset',type =  bool, default = False)
parser.add_argument('--theta', type = float, default = .01)
parser.add_argument('--ntrajectories',type = int, default = 1)
parser.add_argument('--nlearners',type = int, default = 1)
parser.add_argument('--epsilonFactor',type = int, default = 500)
parser.add_argument('--maxNumExp',type = int, default = 500000)
parser.add_argument('--batch_size',type = int, default = 32)
parser.add_argument('--env_id',nargs = '?', default = 'Pong-v0', help = 'Select the environment to run')
parser.add_argument('--obs_scale',type = float, default = 0.5)
parser.add_argument('--resetModel', type = bool, default = False)
parser.add_argument('--nObservations', type = int, default = 4)

args = parser.parse_args()
class Agent:
    def __init__(self,
            session,
            batch_size,
            action_space,
            env,
            epsilon = 0.01,
            greedy = False,
            alpha = 0.1,
            gamma = 0.95,
            visual = True,
            goal = (10,8),
            agentPose = (1,1,'up'),
            showTrial = True,
            randomReset = False,
            epsilonStrat = 1,
            epsilonFactor = 500,
            nEpisodes = 300,
            nObservations = 4,
            maxNumExp = 2000,
            obs_scale = 0.5,
            resetModel = False,
            ob_size = (210,160),
            
            ):

        """
        gridWorld: GridWorld object
        epsilon: value used for epsilon greedy search
        alpha: step size
        gamma: discount favtor
        """
        self.nEpisodes = nEpisodes
        self.resetModel = resetModel
        self.obs_scale = obs_scale
        self.ob_size = self.getOb_size(ob_size,self.obs_scale)
        
        self.session = session
        self.nObservations =nObservations
        self.batch_size = batch_size
        self.epsilonFactor = epsilonFactor
        self.visual = visual
        self.experience = deque()
        self.randomReset = randomReset 
        self.epsilon = epsilon
        self.action_space = action_space
        self.greedy = greedy
        self.epsilonStrat = epsilonStrat
        self.goal = goal
        self.Model = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.exp = []
        self.timeStep = 0
        self.action_space = action_space
        self.env = env
        self.numActions =action_space.n
        self.maxNumExp = maxNumExp


        self.initQValues()
    def getOb_size(self, size, scale):
        if scale !=  1: size = (int(size[0]*scale),int(size[1] * scale))
        return size

    def initQValues(self):
        """
        Output: initializes both model and Q values for all states provided by GridWorld
        """
        self.Qvalues = DQN(self.session, numActions = self.numActions, img_size = self.ob_size, numObservations = self.nObservations)
    def InitState(self,obs):
        e(g('type(obs)'))
        e(g('obs.shape'))
        e(g('self.nObservations'))
        self.currentState = np.stack([obs for i in range(self.nObservations)], axis = 2)
        e(g('self.currentState.shape'))
    def getBestAction(self,pose):
        """
        Pose: (x,y,orientation) tuple representing a pose in grid world
        returns: best action to take dependent on the current Q values
        """
        return self.Qvalues.argmax(self.currentState)
    def sample(self,mylist):
        if len(mylist) == 1: return mylist[0]
        randomInt = random.randint(0,len(mylist) -1)
        return mylist[randomInt]
    def sampleN(self,mylist, n):
        return [self.sample(mylist) for i in range(n)]
    def getAction(self,pose):
        """
        pose: (x,y,pose) tuple representing 
        returns: action selected from epsilon greedy strategy
        """
        if self.epsilonStrat == 1: self.epsilon = self.epsilon
        elif self.epsilonStrat == 2:self.epsilon = self.epsilonFactor /(self.gridWorld.getTotalSteps()+1)
        elif self.epsilonStrat == 3:
            if self.epsilon > .0001:
                self.epsilon -= .0001
        if self.greedy: 
            bestAction = self.getBestAction(pose)
            action = bestAction
            return action
        else:
            p = random.uniform(0,1) 
            if p > self.epsilon:
                action= self.getBestAction(pose)
                
            else: 
                action = random.choice([i for i in range(self.numActions)])
            #e(g('pose'))
            #print(self.Q[pose].getValues())
            e(g('action'))
            return action
    def getNewState(self,observation):
        e(g('self.currentState.shape'))
        e(g('self.nObservations'))
        observation = np.reshape(observation, observation.shape + (1,))
        return np.append(observation, self.currentState[:,:,1:], axis = 2)
    def oneHotActions(self,action):
        actionArray = np.zeros(self.numActions)
        actionArray[action] = 1
        return actionArray
    def updateExp(self,currentState,action, reward,observation, terminal):
        newState = self.getNewState(observation)
        action = self.oneHotActions(action)
        self.experience.append((currentState,action,reward,newState,terminal))
        if len(self.experience) > self.maxNumExp:
                self.experience.popleft()
        self.currentState = newState 
        self.timeStep += 1
    def getGrayScale(self,image):
        imPil = Image.fromarray(image)
        imPil = ImageOps.grayscale(imPil)
        return np.asarray(imPil)
    def preprocess(self,image, scale = 0.5, grayScale = True):
        imPil = Image.fromarray(image)
        if grayScale: imPil = ImageOps.grayscale(imPil)
        if scale != 1: imPil  = imPil.resize((int(imPil.width * scale),int( imPil.height * scale)), Image.BICUBIC)
        return np.asarray(imPil)
    def getBatch(self):
        return self.sampleN(self.experience, self.batch_size)
    def trainOnEnvironment(self):
        for i in range(self.nEpisodes):
            self.env.render()
            ob = self.env.reset()
            ob = self.preprocess(ob, scale = self.obs_scale)
            self.InitState(ob)
            while True:
                if self.visual: self.env.render()
                S = self.currentState
                A = self.getAction(S)
                ob, reward, done ,_ = self.env.step(A)
                ob = self.preprocess(ob, scale = self.obs_scale)
                self.updateExp(S,A,reward,ob,done)
                batch = self.getBatch()
                self.Qvalues.trainOnExperience(batch,self.timeStep)
                if done: break
            
           

                
def main():
    tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    with tf.compat.v1.Session(config = tf_config) as sess:	
        Agents = []
        logger.set_level(logger.INFO)
        env = gym.make(args.env_id)
        outdir = './results/'
        env = wrappers.Monitor(env, directory = outdir, force = True)
        env.seed(0)
        ob = env.reset()
        ob_size = ob.shape

        agent = Agent(
                session = sess,
                batch_size = args.batch_size,
                action_space = env.action_space,
                env = env,
                epsilon =args.epsilon,
                greedy =args.greedy,
                alpha =args.alpha,
                gamma = args.gamma,
                visual =args.visual,
                randomReset =args.randomReset,
                epsilonStrat = args.epsilonStrat,
                epsilonFactor = args.epsilonFactor,
                nEpisodes = args.nEpisodes,
                maxNumExp = args.maxNumExp,
                obs_scale = args.obs_scale,
                resetModel = args.resetModel,
                ob_size = ob_size,
                nObservations = args.nObservations,              
                )
        agent.trainOnEnvironment()
        env.close()

if __name__ == "__main__": 


    main()
#python GridAgent.py --ndirect 15000 --learner 2
