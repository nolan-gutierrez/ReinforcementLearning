class GridVisual:
    def __init__(self, 
            gridWorld,
            ):
        self.gridWorld = gridWorld
    def getAgentVisual(self):
        _,_,o = self.gridWorld.getAgentPose()
        if o == 'up': return 'u'
        elif o == 'down':return 'd'
        elif o == 'left': return 'l'
        elif o == 'right': return 'r'
        else: return 'y'
    def showWorld(self):
        obstacles = self.gridWorld.getObstacles()
        agentPose = self.gridWorld.getAgentPose()
        x1,y1,_ = agentPose
        goal = self.gridWorld.getGoal()
        h,w = self.gridWorld.getHW()
        agent = self.getAgentVisual()
        for y in range(h,0, -1):
            for x in range(1,w + 1):
                if (x,y)  == (x1,y1) :
                    print("", agent, end = " ")
                elif (x,y) == goal:
                    print("",'g', end = " ")
                elif (x,y) in obstacles:
                    print("",'o', end = " ")
                elif x <= 1 or x >= w: 
                    print(" b", end = " ")
                elif y <= 1 or y >= h: 
                    print(" b", end = " ")
                else: print("  ", end = " ")
            print("")
    def getActionMenu(self):

        return "\nw: forward \n d: turn right \n s: backward \n a: turn left \n "
    def getAction(self, action):
        if action == 'w': return 'f'
        elif action == 's': return 'b'
        elif action == 'd': return 'r'
        elif action == 'a': return 'l'
        else: 
            print("Invalid Action, going Forward") 
            return 'f'
