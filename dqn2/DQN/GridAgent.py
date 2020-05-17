from GridWorld import GridWorld
from Debugger import Debugger
from numpy import random
from Counter import Counter
import argparse
from queue import PriorityQueue
from Plotter import Plot
from collections import defaultdict
e,d = exec, Debugger()
e(d.gIS())
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type = float, default = 0.1)
parser.add_argument('--alpha', type = float, default = 0.1)
parser.add_argument('--gamma',type = float, default = 0.95)
parser.add_argument('--visual',type = bool, default = False)
parser.add_argument('--ndirect',type = int, default = 150000)
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
parser.add_argument('--maxNumExp',type = int, default = 2000)

args = parser.parse_args()
class Agent:
    def __init__(self,
            epsilon = 0.01,
            greedy = False,
            alpha = 0.1,
            gamma = 0.95,
            visual = True,
            goal = (10,8),
            agentPose = (1,1,'up'),
            showTrial = True,
            randomReset = False,
            DQN = None,
            DQNBool = False
            epsilonStrat = 1,
            epsilonFactor = 500
            ):

        """
        gridWorld: GridWorld object
        epsilon: value used for epsilon greedy search
        alpha: step size
        gamma: discount favtor
        """
        self.actionValues = Counter()
        self.epsilonFactor = epsilonFactor
        self.randomReset = randomReset 
        self.epsilon = epsilon
        self.greedy = greedy
        self.epsilonStrat = epsilonStrat
        self.goal = goal
        self.Q = dict()
        self.gridWorld = GridWorld(goal,
                agentPose,
                visual = visual,
                showTrial = showTrial,
                randomReset = randomReset)
        self.actions = self.gridWorld.getActions()
        self.Model = dict()
        self.alpha = alpha
        self.PriorityQueue = PriorityQueue()
        self.gamma = gamma
        self.exp = []
        self.rewards = dict()
        self.rewardNums = dict()
        self.predecessors = defaultdict(set)
        self.initQValues()

    def initQValues(self):
        """
        Output: initializes both model and Q values for all states provided by GridWorld
        """
        states = self.gridWorld.getStateSpace()
        for s in states: 
            self.Q[s] = Counter()
            self.rewards[s] = 0
            self.rewardNums[s] = 0
            for a in self.actions:
                news = self.gridWorld.getPoseFromAction(s,a)
                if self.gridWorld.isOutsideBounds(news): news = s
                self.Model[(s,a)] = (0,news)
                self.Q[s].keyValue(a,0)
    def printQValues(self, pose):
        print('actionValues: ', self.Q[pose].getDict())
    def getBestAction(self,pose):
        """
        Pose: (x,y,orientation) tuple representing a pose in grid world
        returns: best action to take dependent on the current Q values
        """
        return self.Q[pose].argmax()
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
            bestAction,_ = self.getBestAction(pose)
            action = bestAction
            return action
        else:
            p = random.uniform(0,1) 
            if p > self.epsilon:
                action,value = self.getBestAction(pose)
                
            else: 
                action = random.choice(list(self.Q[pose].getKeys()))
            #e(g('pose'))
            #print(self.Q[pose].getValues())
            return action
    def getRandomSFromExp(self):
        """
        returns random previously observed state
        """
        if len(self.exp) == 1:
            s,a = self.exp[0]
            return s
        randomInt = random.randint(0,len(self.exp) - 1)
        s,a = self.exp[randomInt]
        return s
    def getRandomAFromRandomS(self,S):
        """
        S: pose (x,y,orientation) tuple representing a pose in grid world
        returns: an action which was previously taken from state S
        """
        aList = []
        for s,a in self.exp:
            if S == s:
                aList += [a]
        return random.choice(aList)
    def getSA(self):
        """
        returns a random state with a random action. Both must have previously happened 
        """
        S = self.getRandomSFromExp()
        A = self.getRandomAFromRandomS(S)
        return S,A
    def updateExp(self,S,A,maxElements):
        """
        args:
            S - tuple of (x,y,orientation)
            A - action
            maxElements - integer, max number of elements to store in experience
        output: 
            - updates experience, where if the number of elements in 
            experience is above maxElements, a random element is removed.
        """
        if len(self.exp) > maxElements:
            randomInt = random.randint(0,len(self.exp) - 1)
            removedElement = self.exp.pop(randomInt)
        self.exp.append((S,A)) 
    def tabDynaQ(self,
            ndirect, # number of iterations for direct learning
            nplanning, # number of iterations for planning
            maxExp, # maximum number of elements to keep in experience
            ):
        """
        learns the Q values using Dyna-Q planning with tabular (deterministic) model
        """
        for i in range(ndirect):
            S = self.gridWorld.getAgentPose()
            A = self.getAction(S)
            self.exp += [(S,A)]
            self.updateExp(S,A,maxExp)
            R,Su = self.gridWorld.takeStep(A)
            Q = self.Q

            delta = R + self.gamma * Q[Su].max() - Q[S].at(A)
            self.Q[S].keyValue( A,   Q[S].at(A) + self.alpha * delta)
            self.Model[(S,A)] = (R,Su)
            for j in range(nplanning):
                S,A = self.getSA()
                R,Su = self.Model[(S,A)]
                delta = R + self.gamma * Q[Su].max() - Q[S].at(A)
                self.Q[S].keyValue( A,   Q[S].at(A) + self.alpha * delta)
    def updateRewards(self,state,R):
        self.rewards[state] += R
        self.rewardNums[state] += 1
    def getRewardAvg(self,state):
        if self.rewardNums[state] == 0 : return 0
        else: return self.rewards[state] / self.rewardNums[state]


    def updatePredecessors(self,S,A,Su):
        self.predecessors[Su].add((S,A))
    def PrioritySweep(self,ndirect,nplanning,theta = 0.1):
        
        for i in range(ndirect):
            S = self.gridWorld.getAgentPose()
            A = self.getAction(S)
            R,Su = self.gridWorld.takeStep(A)
            self.updatePredecessors(S,A,Su)
            self.updateRewards(Su,R)
            self.Model[(S,A)] = (R,Su)
            Q = self.Q
            
            P = abs(R + self.gamma * Q[Su].max() - Q[S].at(A))

            if P > theta: self.PriorityQueue.put((-P,(S,A)))
            iterCount = 0
            while iterCount < nplanning and not self.PriorityQueue.empty():

                P,item = self.PriorityQueue.get()

                S,A = item
                R,Su = self.Model[(S,A)]
                delta = R + self.gamma * Q[Su].max() - Q[S].at(A)
                self.Q[S].keyValue( A,   Q[S].at(A) + self.alpha * delta)
                predStates = self.gridWorld.getPredFromPose(S)
                for SB, AB in predStates:

                    RB,_ = self.Model[(SB,AB)]
                    P = abs(RB + self.gamma * Q[S].max() - Q[SB].at(AB))
                    if P > theta:
                        self.PriorityQueue.put((-P,(SB,AB)))
                iterCount += 1
    def tabDynaQ_Traj(self,
            ndirect, # number of iterations for direct learning
            ntrajectories,
            nplanning,
            numMaxExp# number of iterations for planning
            ):
        """
        args:
            ndirect - number of iterations for direct learning
            ntrajectories - number of individual trajectories to test
            nplanning - number of steps to take for each trajectory
        learns the Q values using Dyna-Q planning with tabular (deterministic) model
        """
        for i in range(ndirect):
            S = self.gridWorld.getAgentPose()
            A = self.getAction(S)
            self.updateExp(S,A,numMaxExp)
            R,Su = self.gridWorld.takeStep(A)
            Q = self.Q

            delta = R + self.gamma * Q[Su].max() - Q[S].at(A)
            self.Q[S].keyValue( A,   Q[S].at(A) + self.alpha * delta)
            self.Model[(S,A)] = (R,Su)
            for t in range(ntrajectories):
                initState =  self.getRandomSFromExp()
                for j in range(nplanning):
                    if j ==0: S =initState
                    A,_ = self.getBestAction(S)
                    
                    R,Su = self.Model[(S,A)]
                    delta = R + self.gamma * Q[Su].max() - Q[S].at(A)
                    self.Q[S].keyValue( A,   Q[S].at(A) + self.alpha * delta)
                    S = Su

    def learn(self,ndirect,nplanning,theta,learner,ntrajectories,maxNumExp):
        if learner == 1: 
            self.tabDynaQ(ndirect,nplanning,maxNumExp)
        elif learner == 2:
            self.PrioritySweep(ndirect,nplanning,theta)
        elif learner ==3:
            self.tabDynaQ_Traj(ndirect,ntrajectories,nplanning,maxNumExp)
def getLearnerName(learner):
    if learner == 1: return "dyna-Q "
    elif learner == 2: return "Prioritized Sweeping "
    else: return "dyna-Q with Trajectory Sampling "
def main():
    trialLists = []
    labels = []
    for i in range(args.nlearners):
        e(g('i'))
        Agents[i].learn(args.ndirect,
                args.nplanning * (i + 1),
                args.theta,
                args.learner,
                args.ntrajectories,
                args.maxNumExp)
        trialList = gridAgents[i].gridWorld.getStepsList()
        trialLists.append(trialList)
        labels.append("nplanning " + str(args.nplanning * ( i + 1)))
    plotter = Plot()
    plotter.plotListGeneral(trialLists, 
            labels,
            'Trial #',
            'TrialSteps to Reach Goal',
           getLearnerName(args.learner) + str(args.ndirect) + ' steps')

if __name__ == "__main__": 
    Agents = []
    for i in range(args.nlearners):
        Agents += [Agent(args.epsilon,
                args.greedy,
                args.alpha,
                args.gamma,
                args.visual,
                randomReset =args.randomReset,
                epsilonStrat = args.epsilonStrat,
                epsilonFactor = args.epsilonFactor)]
    main()
#python GridAgent.py --ndirect 15000 --learner 2
