from Environment import Environment as env
from numpy import random
from Counter import Counter
class ElevatorAgent: 
    def __init__(self, floor = 1, epsilon = 0.1, greedy = False):
        self.floor = floor
        self.actionValues = Counter()
        self.actionNumRewards = {}
        self.epsilon = epsilon
        self.greedy = greedy
    def setFloor(self,floor):
        self.floor = floor
    def getFloor(self):
        return self.floor
    def moveUP(self):
        self.floor += 1
    def moveDown(self):
        self.floor -= 1
    def getQValues(self):
        return self.actionValues.getDict().copy()
    def initQ(self, actions):
        for i in actions:
            self.actionValues.keyValue(i,0)
        for i in actions:
            self.actionNumRewards[i] = 0
    def pAF(self):
        print('actionValues: ', self.actionValues.getDict())
        print('actionNumRewards: ', self.actionNumRewards)
    def getBestAction(self):
        #This function returns the best action based on 
        #current values
        return self.actionValues.argmax()
    def getAction(self):
        if self.greedy: 
            bestAction,_ = self.getBestAction()
            return bestAction
        else:
            p = random.uniform(0,1) 
            if p > self.epsilon:
                bestAction,_ = self.getBestAction()
                return bestAction
            else: return random.choice(list(self.actionValues.getKeys()))
    def takeAction(self):
        self.setFloor(self.getAction())
    def updateActionValue(self,
            a, #action
            r, #reward
            step = None, #stepsize
            ):
        Q = self.actionValues
        # increment reward number
        self.actionNumRewards[a] +=1
        k = self.actionNumRewards[a]
        l= 1/k if not step else step
        #incremental difference formula
        Q.keyValue(a, Q.at(a) + l * (r - Q.at(a)))



