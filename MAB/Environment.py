import math
from Debugger import Debugger
from Probability import ElevatorProb as EP
e,d = exec, Debugger()
e(d.gIS())
class Environment:
    def __init__(self,
            elevatorAgent,
            ep,
            numFloors = 6,
            episodeLength = 100,
            timeStep = 7,
            step = None,
            squaredPenalty = False,

            ):
        # Takes a probability distribution for the call floors 
        # and exit floors. 
        self.step = step
        self.squaredPenalty = squaredPenalty
        self.a = elevatorAgent
        self.numFloors = numFloors
        self.callBool = False
        self.exitBool = False
        self.callFloor = None
        self.exitFloor = None
        self.justFinished = False
        self.Time = 0
        self.T = episodeLength
        self.timeStep = 7
        self.startOfNextEp = episodeLength
        self.episodeNumber = 1
        self.a.initQ(self.getArms())
        self.ep = ep
    def pF(self):
        print('callFloor: ', self.callFloor)
        print('exitFloor: ', self.exitFloor)
        print('callBool: ', self.callBool)
        print('exitBOol: ', self.exitBool)
        print('justFinished: ', self.justFinished)
        print('agentFloor: ', self.a.getFloor())
        print('')
    def pT(self):
        print('Time: ' , self.Time)
        print('timeStep: ', self.timeStep)
        print('startOfNextEp: ', self.startOfNextEp)
        print('episodeNumber: ', self.episodeNumber)
        print('')
    def pQV(self):
        print('actionValues: ', self.a.getQValues())
        print('bestAction: ', self.a.getBestAction())
        print('')

    def setFloorInfo(self, 
            callFloor,
            exitFloor,
            callBool,
            exitBool,
            justFinished,
            ):
        self.callFloor = callFloor
        self.exitFloor = exitFloor
        self.callBool = callBool
        self.exitBool = exitBool
        self.justFinished = justFinished
    def getTimeToFinish(self, a, c,e):
        # takes a floor, calling floor and an exit floor
        return -7 * (abs(a-c) + abs(c - e) + 1)
    def getUtility(self, a,c,e):
        T = self.getTimeToFinish(a,c,e)
        if self.squaredPenalty:
            return -(T ** 2)
        else: return T

    def getArms(self): 
        #returns a list of actions 
        return [i for i in range(1,self.numFloors + 1)]
    def moveElevator(self):
        # This function moves the elevator up if the calling 
        # elevator is above and down otherwise. It acts similarly
        # for the exitFloor. 

        a = self.a
        if self.callBool: 
            if self.callFloor > a.getFloor(): a.moveUP()
            elif self.callFloor < a.getFloor(): a.moveDown()
            if a.getFloor() == self.callFloor:
                self.callBool = False
                self.exitBool = True
        elif self.exitBool: 
            if self.exitFloor > a.getFloor():
                a.moveUP()
            elif self.exitFloor < a.getFloor():
                a.moveDown()
            if a.getFloor() == self.exitFloor: 
                self.exitBool = False
                self.justFinished = True
    def updateTime(self):
        if self.Time >= self.startOfNextEp:
            self.justFinished = True
        if self.justFinished: 
            self.Time = self.startOfNextEp
            self.startOfNextEp += self.T
            self.episodeNumber += 1 
            self.justFinished = False
        else: self.Time += self.timeStep
        
    def getPerson(self):
        # This function updates the action value based on the 
        # reward obtained from picking up a person on callFloor
        # and dropping that person off at exit floor
        # This function should only be called when callfloor 
        # and exit floor have been set.

        #get reward
        reward = self.getUtility(self.a.getFloor(),self.callFloor,self.exitFloor)
        #update time by length of episode
        self.Time += self.timeStep
        #update episode num
        self.episodeNumber +=1
        #update q values
        self.a.updateActionValue(self.a.getFloor(), reward, self.step)
        #sets floor for next round
        self.a.takeAction()
    def update( self):
        self.callFloor = self.ep.sampleCall()
        self.exitFloor = self.ep.sampleExit(self.callFloor)
        self.getPerson()


        
            



