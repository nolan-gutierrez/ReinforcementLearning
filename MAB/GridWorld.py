import numpy as np
from numpy import random as r
from Debugger import Debugger 
from GridVisual import GridVisual
e,d = exec, Debugger()
e(d.gIS())
class GridWorld:
        def __init__(self,
                goal,
                agentPose,
                obstacleList = [(12,13), (5,6)],
                height = 15,
                width = 25,
                visual = True,
                showTrial = True,
                randomReset = False):
            self.grid = np.zeros((height,width))
            # forward, backward, left, right
            self.showTrial = showTrial
            self.randomReset = randomReset
            self.actions = ['f','b','l','r']
            self.orientations = [ 'up','down','left','right']
            self.obstacles = obstacleList
            self.goal = goal
            self.agentPose = agentPose
            self.initialPose = agentPose
            self.OWeights = [.9,.1]
            self.PWeights = [.8,.2]
            self.results = ['success', 'fail']
            self.trialReset = True
            self.trialNumber = 0
            self.trialSteps = 0
            self.totalSteps = 0
            self.stepsList = []
            self.height = height
            self.width = width
            self.visual = visual
            if self.visual:
                self.gridVisual = GridVisual(self)

            
        def getGridVisual(self):
            return self.gridVisual
        def getTotalSteps(self):
            return self.totalSteps
        def getHW(self):
            return self.height,self.width
        def getActions(self):
            return self.actions
        def getGoal(self):
            return self.goal
        def getOrientations(self):
            return self.orientations
        def turnAgent(self,action,pose, sample = True):
            """ 
            Parameters: 
            action: action to take in form of a string
            returns: new state with different orientation but same position

            """
            x,y,o = pose
            if sample:
                sampleResult = r.choice(self.results,p =self.OWeights)
                if sampleResult == 'fail': return (x,y,o)

            if action == 'l': 
                if o == 'up': return (x,y,'left')
                if o == 'left': return(x,y,'down')
                if o == 'down' : return (x,y, 'right')
                if o == 'right': return (x,y,'up')
            if action == 'r': 
                if o == 'up': return (x,y,'right')
                if o == 'right': return (x,y, 'down')
                if o == 'down' : return (x,y,'left')
                if o == 'left': return (x,y,'up')
        def moveAgent(self,action,pose,sample = True):
            """ 
            Parameters:
            Action: action to take in form of string
            returns: new state with different position but same orientation


            """
            x,y,o = pose
            if sample:
                sampleResult = r.choice(self.results,p =self.PWeights)
                if sampleResult == 'fail': return (x,y,o)
            S = [ (x,y + 1,o), (x,y-1,o), (x-1,y,o), (x+1,y,o)]
            if action == 'f':

                if o == 'up': 
                    return S[0]
                if o == 'down': return S[1]
                if o == 'left': return S[2]
                if o == 'right': return S[3]
            if action == 'b':
                if o == 'up': return S[1]
                if o == 'down': return S[0]
                if o == 'left': return S[3]
                if o == 'right': return S[2]
        def isOutsideBounds(self,pose):
            """
            Parameters: 
            pose: (x,y,o)
            returns: True if position is outside grid, false otherwise
            """
            x,y,_ = pose
            if x > self.width or x < 1 or y < 1 or y > self.height: return True
            elif (x,y) in self.obstacles: return True
            else: return False
        def getPoseFromAction(self,pose,action):

            if action == 'f' or action =='b': 

                newPose = self.moveAgent(action,pose)
            if action == 'l' or action == 'r':
                newPose = self.turnAgent(action,pose)
            return newPose
        def getReward(self, pose):
            x,y,o = pose
            """ 
            Parameters: None
            returns reward

            """
            if (x,y) == self.goal: return 100
            elif (x,y) in self.obstacles: return -100
            else: return 0 
        def getAgentOrientation(self):
            _,_,o = self.agentPose
            return o
        def getAgentPosition(self):
            x,y,_ = self.agentPose
            return (x,y)
        def getAgentPose(self):
            return self.agentPose
        def getObstacles(self):
            return self.obstacles
        def setPose(self,pose):
            """
            Parameters: 
            pose: tuple of (x,y,orientation)
            output: changes pose of the agent in world
            """
            self.agentPose = pose
        def isTerminalState(self,pose):
            """
            args:
                pose - (x,y,orientation) tuple
            returns: True if pose is terminal state and false otherwise.
            """
            x,y,o = pose
            if (x,y) == self.goal:
                return True
            else:
                return False
        def getTrialReset(self):
            return self.trialReset
        def getStepsList(self):
            """
            returns: list of steps it takes to reach goal state 
            for each trial
            """
            return self.stepsList
        def updateTrialInfo(self, successorState):
            """
            args:
                successorState - (x,y,orientation) tuple
            Updates trial info. If next state is terminal state then
            we should reset number of steps for current trial and 
            we should add to a list of steps that is for each trial.
            """
            if self.isTerminalState(successorState):
                self.trialNumber += 1
                if self.trialNumber % 10 == 0 and self.showTrial:
                    e(g('self.stepsList'))
                    e(g('self.trialNumber'))

                self.stepsList += [self.trialSteps]
                self.trialSteps = 0
                self.trialReset = True
            self.trialSteps += 1
            self.totalSteps +=1
        def initIfReset(self):
            """
            Randomly resets agent pose if trialReset boolean is set to true
            """
            if self.trialReset:
                states = self.getStateSpace()
                if self.randomReset: newState = self.getRandomState()
                else: newState = self.initialPose 
                self.agentPose = newState
                self.trialReset = False

        def getRandomState(self):
            states = self.getStateSpace()
            randomInt = r.randint(0,len(states) - 1)
            return states[randomInt]

        def getRSu(self,S,A):
            Su = self.getPoseFromAction(S,A)
            if self.isOutsideBounds(Su):
                return self.getReward(Su),S
            else: return self.getReward(Su),Su
        def takeStep(self,action):
            """
            This function should maintain information on whether this 
            state a terminal state. After obtaining the successor state,
            the successor state is checked to see if it is a terminal 
            state. Only the goal is a successor state. 
            Parameters:
            action: takes an action as input e.g. 'up'
            returns: reward for agent, successor state
            """
            self.initIfReset()    
            currentState = self.agentPose
            reward,successorState = self.getRSu(currentState,action)
            self.agentPose = successorState
            self.updateTrialInfo(successorState)
            if self.visual: self.gridVisual.showWorld() 
            return (reward,successorState) # (R,S')
        def getPredFromPose(self,pose):
            """
            args: 
                pose - (x,y,orientation) tuple
            returns - possible predecessor states of pose given possible
            actions. Only states not in obstacles or boundaries are allowed
            """
            x,y,o = pose
            if o == 'left':
                predStates = [((x + 1,y, 'left'), 'f'),((x,y,'up') ,'l'),((x,y,'down'), 'r'),((x-1,y,'left'),'b')]
            elif o == 'right':
                predStates = [((x - 1,y, 'right'), 'f'),((x,y,'up') ,'r'),((x,y,'down'), 'l'),((x+1,y,'right'),'b')]
            elif o == 'up':
                predStates = [((x ,y-1, 'up'), 'f'),((x,y,'right') ,'l'),((x,y,'left'), 'r'),((x,y + 1,'up'),'b')]
            else: 
                predStates = [((x ,y+1, 'down'), 'f'),((x,y,'right') ,'r'),((x,y,'left'), 'l'),((x,y-1,'down'),'b')]
            validPredStates = []
            for predState in predStates:
                if not self.isOutsideBounds(predState[0]):
                    validPredStates.append(predState)
            return validPredStates

        def getStateSpace(self):
            states = []
            for y in range(1,self.height+ 1):
                for x in range(1,self.width+1):
                    for o in self.orientations:
                        states += [(x,y,o)]
            return states
        def addObstacle(self,newObstacle):
            """
            Parameters:
            newObstacle: new obstacle of form (x,y)
            output: adds new obstacle to environment
            """
            self.obstacles += [newObstacle]
        def takeStepExperience(self,action):
            """
            Parameters:
            action: takes an action as input e.g. 'up'
            returns: (S, A, S', R) 
            """
            S = self.agentPose
            A = action
            R,Su = self.takeStep(A)
            return (S,A,Su,R)
