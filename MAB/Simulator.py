from Elevator import ElevatorAgent
from Environment import Environment
from Plotter import Plot
class ElevatorSim:
    def __init__(self,
            agent,
            ep,
            env,
            step = None,
            numFloors = 6 ):
        self.step = step
        self.agent = agent
        self.env = env
        self.QHis = [self.agent.getQValues()]
        #saves preliminary history of 0
    def getQHis(self):
        return self.QHis
    def runForIterations(self, numIterations, plot = True):
        for i in range(numIterations):
           self.env.update()
           self.QHis.append(self.agent.getQValues())
        if plot:
            plt = Plot() 
            plt.plotList(self.QHis)
        

