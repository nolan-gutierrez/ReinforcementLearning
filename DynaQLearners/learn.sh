d:
cd D:/nbgbl/Documents/GITrep/ReinforcementLearning/MAB
conda activate tf
python  GridAgent.py --ndirect 100000 --epsilon 0.01 --learner 1 --nlearners 3 --epsilonStrat 2 --nplanning 3 --maxNumExp 500
python  GridAgent.py --ndirect 100000 --epsilon 0.01 --learner 2 --nlearners 3 --epsilonStrat 1 --epsilonFactor 500 --nplanning 3 --alpha 0.1 --theta 0.1
python  GridAgent.py --ndirect 100000 --epsilon 0.01 --learner 3 --nlearners 3 --epsilonStrat 2 --nplanning 3 --maxNumExp 500