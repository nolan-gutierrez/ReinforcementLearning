
from PIL import ImageOps,Image
from collections import deque
import pickle
from Debugger import Debugger
from random import sample
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
#import pybulletgym
import gym
e,d = exec, Debugger()
e(d.gIS())
d.sDS(False)
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type = float, default = 0.05)
parser.add_argument('--alpha', type = float, default = 0.1)
parser.add_argument('--gamma',type = float, default = 0.99)
parser.add_argument('--visual',type = bool, default = False)
parser.add_argument('--nEpisodes',type = int, default = 600)
parser.add_argument('--nplanning', type = int, default = 10)
parser.add_argument('--greedy', type = bool, default = False)
parser.add_argument('--epsilonStrat',type = int, default = 5)
parser.add_argument('--learner', type = int, default =1)
parser.add_argument('--showTrial', type = bool, default = True)
parser.add_argument('--randomReset',type =  bool, default = False)
parser.add_argument('--theta', type = float, default = .01)
parser.add_argument('--ntrajectories',type = int, default = 1)
parser.add_argument('--nlearners',type = int, default = 1)
parser.add_argument('--epsilonFactor',type = int, default = 500)
parser.add_argument('--maxNumExp',type = int, default = 300000)
parser.add_argument('--batch_size',type = int, default = 32)
parser.add_argument('--env_id',nargs = '?', default = 'PongDeterministic-v4', help = 'Select the environment to run')
parser.add_argument('--obs_scale',type = float, default = 0.525)
parser.add_argument('--resetModel', type = bool, default = False)
parser.add_argument('--nObservations', type = int, default = 4)
parser.add_argument('--replayType', type = int, default = 1)
parser.add_argument('--expBool', type = bool, default = True)
parser.add_argument('--timeToStart', type = int ,default = 10000)
parser.add_argument('--cUpdate', type = int, default = 10000)
parser.add_argument('--saveTime', type = int , default = 30000)
parser.add_argument('--timeToSaveExp', type = int, default = 50000)
parser.add_argument('--resetExp', type = bool, default = False)
parser.add_argument('--trainUpdate', type = int, default = 4)
parser.add_argument('--testPhase', type = bool, default = False)
parser.add_argument('--checkpointName', type = str, default = None)


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
            epsilonStrat = 4,
            epsilonFactor = 500,
            nEpisodes = 300,
            nObservations = 4,
            maxNumExp = 2000,
            obs_scale = 0.5,
            resetModel = False,
            ob_size = (210,160),
            replayType = 2,
            expBool = True,
            timeToStart = 2000,
            cUpdate = 10000,
            saveTime = 10000,
            timeToSaveExp = 20000,
            resetExp = False,
            trainUpdate = 4,
            testPhase = False,
            checkpointName = None,
            
            ):

        """
        Params:
            - session: tensorflow session 
            - batch_size: size of minibatch
            - action_space: action space provided by the environment
            - env: gym environment
            - epsilon: proportion of time that random actions are taken
            - greedy: Whether to use greedy exploration or not; boolean
            - alpha: not used for gym
            - gamma: discount factor
            - visual: boolean, true if rendering environment
            - goal: not used for gym
            - agentPose: not used for gym
            - showTrial: Whether to show the current trial. 
            -randomReset: not used for gym
            - epsilonStrat: choice between 1-5
                - 1: epsilon greedy exploration
                - 2: epsilong greedy exploration where epsilon decreases by formula epsilon = epsilonFactor/stepsize
                - 3: linear decreasing epsilon strategy
                - 4: random exploration
                - 5: random exploration for duration timeToStart
            -epsilonFactor: used for formula of epsilon greedy strategy 2
            - nEpisodes: num episodes to train for
            - nObservations: number of observations in history
            -maxNumExp: max numbe of experiences in replay buffer
            - obs_scale: factor to scale observations b y 
            - replayType: 1 for normal experience replay, 2 for combined experience replay
            - expBool : whether to collect experience
            -timeToStart : time to start training. Also used for exploration strategy 5
            - cUpdate: time between target network updates
            - saveTime: steps betweens saving models. 
            - timeToSaveExp: When to save Exp
            - resetExp: If to not load old Experience 
            - trainUpdate: How  many steps in between updating parameters
            - testPhase: If to  train or not. 
            - checkpointName: full path to name of checkpoint. directory of checkpoint should include checkpoint file.
        output: stores all parameters, creates target and main q-value networks. 

        """
        self.env = env
        self.checkpointName = checkpointName
        self.testPhase = testPhase
        self.trainUpdate = trainUpdate
        self.timeToSaveExp = timeToSaveExp
        self.cUpdate = cUpdate
        self.timeToStart = timeToStart
        self.saveTime = saveTime
        self.nEpisodes = nEpisodes
        self.resetExp = resetExp
        self.replayType = replayType
        self.expBool = expBool
        self.resetModel = resetModel
        self.obs_scale = obs_scale
        self.ob_size = self.getOb_size()
        
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
        self.numActions =4
        self.maxNumExp = maxNumExp
        self.episodeRewards = []
        self.rewards = []


        self.initQValues()
    def getOb_size(self):
        """
        output: Gets example observation from environment to determine scale for use in the DQN
        """
        ob = self.env.reset()
        ob = self.preprocess(ob, self.obs_scale, True)
        size = ob.shape
        return size
    def loadExp(self):
        """
        output: loads replay buffer into self.experience
        """
        try:
            with open('mylist','rb') as f: 
                self.experience = pickle.load(f)
        except: print("file not found")
        
    def saveExp(self): 
        """
        output: saves experience in self.experience to mylist file
        """
        print("experience saved")
        with open('mylist','wb') as f:
            pickle.dump(self.experience,f)
    def saveRewards(self):
        """
        output: saves list of rewards. Rewards can be directly passed to methods of Plotter object

        """
        with open('rewards','wb') as f:
            pickle.dump(self.episodeRewards,f)

    def initQValues(self):
        """
        Output: initializes both model and Q values for all states provided by GridWorld
        """
        self.Qvalues = DQN(self.session,gamma = self.gamma, numActions = self.numActions, img_size = self.ob_size, numObservations = self.nObservations,timeToStart = self.timeToStart, cUpdate = self.cUpdate, saveTime = self.saveTime, checkpointName = self.checkpointName, resetModel = self.resetModel)

    def InitState(self,obs):
        self.currentState = np.stack([obs for i in range(self.nObservations)], axis = 2)
    def getBestAction(self,pose):
        """
        Params: 
            - returns optimal action of Qvalues using argmax
        returns: best action to take dependent on the current Q values
        """
        return self.Qvalues.argmax(self.currentState)
    def sample(self,mylist):
        """
        Params:
            - mylist - list
        returns: random element from mylist
        """
        if len(mylist) == 1: return mylist[0]
        randomInt = random.randint(0,len(mylist) -1)
        return mylist[randomInt]
    def sampleN(self,mylist, n):
        """
        Params:
            mylist - list
            n - number of random items to sample with replacement
        returns: n random items from mylist
        """
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
        elif self.epsilonStrat == 4 or 5: 
            if self.epsilonStrat == 5:
                if self.timeStep > self.timeToStart:
                    self.epsilonStrat = 1
            return  random.choice([i for i in range(self.numActions)])
        

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
        """
        Params: 
            observation - input observation from environment
        returns: past four observations as an array
        """
        stateCopy = self.currentState.copy()
        stateCopy[:,:,1:4] = self.currentState[:,:,1:4]
        stateCopy[:,:,0] = observation
        return stateCopy
    def oneHotActions(self,action):
        """
        Params:
            action - integer value to be one hot encoded 
        returns: array of zeros where array at action is 1
        """
        actionArray = np.zeros(self.numActions)
        actionArray[action] = 1
        return actionArray
    def updateExp(self,currentState,action, reward,observation, terminal,expBool):
        """
        Params: 
            currentState - 84 by 84 by 4 current stateCopy
            action - integer value representing action taken from currentState
            reward - reward received after taking action from currentState
            observation - observation received from environment
            terminal - boolean representing whether current state is terminal
            expBool - boolean that's true if experience should be being updated
        returns - most recent exp
        """
        
        newState = self.getNewState(observation)
        action = self.oneHotActions(action)
        self.experience.append((currentState,action,reward,newState,terminal))
        if len(self.experience) > self.maxNumExp:
                self.experience.popleft()
        recentExp = (currentState, action, reward, newState, terminal)
        self.currentState = newState 
        return recentExp
    def getGrayScale(self,image):
        """
        Params:
            image - image array rgb valued
        returns grayscale image
        """
        imPil = Image.fromarray(image)
        imPil = ImageOps.grayscale(imPil)
        return np.asarray(imPil)
    def preprocess(self,image, scale = 0.525, grayScale = True,binaryThresh = True):
        """
        Params: 
            image - rgb image from environment
            scale - scale to apply to each image
            grayscale - boolean deciding whether to change image to grayscale
            binaryThresh - unused
        """
        #imPil = Image.fromarray(image)
        img = image
        img = np.reshape(img,(210,160,3)).astype(np.uint8)
        if grayScale:

            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            #imPil = ImageOps.grayscale(imPil)
        img = cv2.adaptiveThreshold(img.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        
        if scale != 1:
            img = Image.fromarray(img)
            img  = img.resize((int(img.width * scale),int( img.height * scale)), Image.BILINEAR)
            img = np.asarray(img)

        imArray = np.asarray(img)
        #if binaryThresh: _,imArray = cv2.threshold(imArray, 1, 255, cv2.THRESH_BINARY_INV)
        imArray = imArray[18:102,:]
        return imArray.astype(np.uint8)
    def getBatch(self):
        """
        returns batch
        """
        return self.sampleN(self.experience, self.batch_size)
    def showImageArray(self,image):
        """
        Params: 
            image - image array
        output: shows image array
        """
        im = Image.fromarray(image)
        im.show()
    def clip_reward(self,reward):
        """
        Params: 
            reward - float reward
        returns: clipped reward
        """
        if reward > 0: 
            return 1
        elif reward == 0: 
            return 0
        else: return -1
    def trainOnEnvironment(self):
        if not self.resetExp:self.loadExp()
        for i in range(self.nEpisodes):
            
            ob = self.env.reset()
            ob = self.preprocess(ob, scale = self.obs_scale)
            if i %5 == 0: 
                print("timeStep: ",self.timeStep)
                print("episode: ",i)
                print("len(self.experience):", len(self.experience))
                #self.showImageArray(ob)
                self.env.render()
                averageReward = np.sum(self.rewards)/5
                self.episodeRewards.append(averageReward)
                self.rewards = []
                print("averageReward:, " , averageReward)
            self.InitState(ob)
            while True:
                if self.visual: self.env.render()
                S = self.currentState
                A = self.getAction(S)
                ob, reward, done ,_ = self.env.step(A)
                reward = self.clip_reward(reward)
                self.rewards.append(reward)

                recentExp = None
                if self.expBool:
                    ob = self.preprocess(ob, scale = self.obs_scale)

                    recentExp = self.updateExp(S,A,reward,ob,done,self.expBool)
                if self.timeStep % self.trainUpdate == 0:
                    if not self.testPhase:
                        batch = self.getBatch()
                        self.Qvalues.trainOnExperience(batch,self.timeStep, recentExp, self.replayType)
                if self.timeStep % self.timeToSaveExp ==0 :
                    if len(self.experience )<self.maxNumExp:
                        if self.timeStep is not 0:
                            self.saveExp()
                self.timeStep+=1

                if done:
                    
                    break
        self.saveRewards()  
            
           

                
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
                replayType = args.replayType,
                expBool = args.expBool,
                timeToStart = args.timeToStart,
                cUpdate = args.cUpdate,
                saveTime = args.saveTime,
                timeToSaveExp = args.timeToSaveExp,
                resetExp = args.resetExp,
                trainUpdate = args.trainUpdate,
                testPhase = args.testPhase,
                checkpointName = args.checkpointName
                )
        agent.trainOnEnvironment()
        env.close()

if __name__ == "__main__": 


    main()
#python GridAgent.py --ndirect 15000 --learner 2
