Nolan Gutierrez
UTA ID: 1001033225
Homework Assignment 1
Reinforcement Learning


This code for this homework is divided into 7 main files. The file that 
executes the program can be seen in test.py. The file test.py runs the code by first
initializing a simulator, an environment, an elevator probability object, and
an agent. The simulator then runs  with the default number of iterations as
500. In order to change the number of iterations the user can type
--iterations as an optional parameter when executing test.py. The program
has the following requirements:

python version 3.7.6
numpy 
matplotlib
Download the zip and extract the contents into a place of your choice. 
To run the program for the problem as is, the following command can be 
entered from the command line at the directory where each of the files are
stored:
python test.py
To run the program for 1d) where the agent takes a random action 50% of the time, 
run the following command:
python test.py --epsilon 0.5

For 1d) of the assignment I chose three types of distributions to test my 
learning agent on. The commands that are shown use an epsilon of 0.5, but
"--epsilon 0.5" can be omitted to run the program with the default epsilon of
0.1  The distributions and their commands are as follows:

call - Poisson, exit - Poisson
python test.py --epsilon 0.5 --callPoisson True --exitPoisson True

call - Gaussian with mean 0, exit floor fixed at 1
python test.py --epsilon 0.5 --callGaussian True --exitFloor 1

call - set at 4, exit floor follows Poisson distribution with mean 2
python test.py --epsilon 0.5 --callFloor 4 --exitPoisson True

2a) 
To obtain the results of the learning agent for a penalty that depends
quadratically on the waiting time, run these commands separately for each of
the four scenarios as shown in order on the written report. 
python test.py --epsilon 0.5 --squaredPenalty True
python test.py --epsilon 0.5 --callPoisson True --exitPoisson True --squaredPenalty True
python test.py --epsilon 0.5 --callGaussian True --exitFloor 1 --squaredPenalty True
python test.py --epsilon 0.5 --callFloor 4 --exitPoisson True --squaredPenalty True




