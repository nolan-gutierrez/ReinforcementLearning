import matplotlib.pyplot as plt
class Plot:
    def __init__(self,numFloors = 6):
        self.numFloors = numFloors
        self.floors = [ i for i in range(1,numFloors + 1)]
    def getFloors(self):
        # returns list of floor strings to be used for plotting
        myFloors = []
        for i in range(1,self.numFloors + 1):
            myFloors.append('Floor ' + str(i))
        return myFloors
    def getListsForPlot(self,myList):
        floorsValues = {}
        for i in range(1,self.numFloors + 1):
            # {1:[], 2:[], ... 6:[]}
            floorsValues[i] = []
        for i in range(len(myList)):
            for j in range(1,self.numFloors + 1):
                floorsValues[j].append(myList[i][j])
        return floorsValues
        # Floorvalues should be 6 lists of values. One for each floor. 
        # Each of these will be plotted for against the number of iterations
        # which in this case will be 500
    def plotList(self,myList):
        totalTime = len(myList) # 500 default
        floorValues = self.getListsForPlot(myList)
        floorNames = self.getFloors()
        lineHs = {}
        for i in self.floors:
            lineHs[i] = plt.plot(floorValues[i], label = floorNames[i -1])
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.title('Action Utility over Iterations')
        plt.legend()
        plt.show()
