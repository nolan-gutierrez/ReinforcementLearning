from GridWorld import GridWorld as G
from GridVisual import GridVisual
from Debugger import Debugger 
e,d = exec, Debugger()
e(d.gIS())
goal = (5,5)
agentPose = (8,4,'up')

obstacle = (2,5)

newGrid = G(goal,agentPose)
e(g('newGrid.getObstacles()'))
newGrid.addObstacle(obstacle)
e(g('newGrid.getObstacles()'))
actions = newGrid.getActions()
e(g('actions'))
e(g('newGrid.getAgentPosition()'))
e(('newGrid.getAgentOrientation()'))

preds = newGrid.getPredFromPose(agentPose)
for pred in preds:
    e(g('pred[0][0:2]'))
    newGrid.addObstacle(pred[0][0:2])
gridVisual = newGrid.getGridVisual()
gridVisual.showWorld()
