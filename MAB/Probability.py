from numpy import random as r
from Counter import Counter
import numpy as np
class ElevatorProb:
    def __init__(self,
            numFloors = 6,
            callUniform = True,
            exitUniform = False,
            exitGaussian = False,
            callGaussian = False,
            exitPoisson = False,
            callPoisson = False,
            callFloor = None,
            exitFloor = None
            ):
        self.numFloors = numFloors
        self.callUniform = callUniform
        self.callGaussian = callGaussian
        self.exitUniform = exitUniform
        self.exitGaussian = exitGaussian
        self.callPoisson = callPoisson
        self.exitPoisson = exitPoisson
        self.callFloor = callFloor
        self.exitFloor = exitFloor
        self.arms = [i for i in range(1,numFloors + 1)]
    def sampleCall(self, weights = None):
        # This function samples from a prior distribution
        # with a uniform distribution or the weights provided
        # by the user. 
        if self.callUniform: return r.choice(self.arms)
        elif self.callGaussian: return self.getGaussianAction()
        elif self.callPoisson: return self.getPoissonAction()
        if self.callFloor is not None: return self.callFloor
        else: return r.choice(self.arms,p = weights)
    def getProporCounter(self,callFloor):
        #This function allocates probabilities to each exit 
        # floor conditioned on the callFloor based on what was
        # specified in 1d
        counter = Counter()
        counter.initKeys(self.arms)
        value = 0
        key = callFloor
        for i in range(1,callFloor +1):
            counter.keyValue(key,value)
            key -=1
            value +=1 
        for i in range(callFloor + 1, self.numFloors + 1):
            counter.keyValue(i,5)
        return counter
    def getGaussianAction(self, mean = 3, sigma = 3):
        actionFloat = r.normal(mean,sigma,(1,))
        actionClip = np.clip(actionFloat,1,self.numFloors)
        actionInt = int(actionClip)
        return actionInt
    def getPoissonAction( self, mean = 2):
        action = r.poisson(mean)
        actionClip = np.clip(action,1,self.numFloors)
        return action
        
    def sampleExit(self,callFloor):
        #This function uses the probability distribution
        #provided by 1d for sampling from an exit given 
        # what the call floor is. This function can calculate
        # the probability of an exit floor given what the call
        # floor is, and then sample accordingly.

        #The function works as follows: if a person calls from
        # floor 3 then the following table represents the 
        #probability of exiting from each floor:
        # {1:2, 2: 1, 3: 0, 4: 5,5: 5, 6:5} 
        # We can normalize to find the probability of 
        if self.exitUniform: return r.choice(self.arms) 
        if self.exitGaussian: return self.getGaussianAction()
        if self.exitPoisson: return self.getPoissonAction()
        if self.exitFloor is not None: return self.exitFloor
        exitCounter = self.getProporCounter(callFloor)
        exitCounter = exitCounter.normalize(copy = False)
        actions, probabilities = exitCounter.getItemsAsLists()
        return r.choice(actions,p =probabilities)


        

        
