from Environment import Environment as E
from Elevator import ElevatorAgent as A
from Simulator import ElevatorSim
from Probability import ElevatorProb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type = float,default = 0.1)
parser.add_argument('--greedy', type = bool, default = False)
parser.add_argument('--squaredPenalty', type = bool, default = False)
parser.add_argument('--plot', type = bool, default = True)
parser.add_argument('--iterations', type = int, default = 500)

parser.add_argument('--exitGaussian', type = bool, default = False)

parser.add_argument('--callGaussian', type = bool, default = False)
parser.add_argument('--callUniform', type = bool, default = False)
parser.add_argument('--exitUniform', type = bool, default = False)
parser.add_argument('--exitPoisson', type = bool, default = False)
parser.add_argument('--callPoisson', type = bool, default = False)
parser.add_argument('--callFloor', type = int, default = None)
parser.add_argument('--exitFloor', type = int, default = None)
args = parser.parse_args()
def main():    
    agent = A(epsilon = args.epsilon, greedy = args.greedy)
    ep = ElevatorProb(callUniform = args.callUniform, 
            exitUniform =args.exitUniform,
            exitGaussian = args.exitGaussian,
            callGaussian = args.callGaussian,
            exitPoisson = args.exitPoisson,
            callPoisson = args.callPoisson,
            callFloor = args.callFloor,
            exitFloor = args.exitFloor) 
    env = E(agent,ep,squaredPenalty = args.squaredPenalty)
    sim = ElevatorSim(agent, ep,env)
    sim.runForIterations(args.iterations, plot = args.plot)
if __name__ == "__main__":
    main()

