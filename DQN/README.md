## Simple DQN
As you will see with this repository, this is one of the most simple implementations
of DQN there is online. 

Playing Pong with DQN and Combined Experience Replay
The agent was trained on an intel 8700k cpu with a P4000 GPU with 32 GB RAM. 
The agent took only three hours to reach its level of performance.  In order 
to use the agent, it is recommended to install Anaconda 3 on Ubuntu 16.04 although 
any Ubuntu version should work. The deep learning API used was tensorflow v2, 
and the required libraries are gym, matplotlib, pillow, opencv, ffmpeg, and 
pybullet-gym. The anaconda environment should be made as follows:
```
conda create -y -n gym2 pip python=3.7
conda activate gym2
conda install -y tensorflow matplotlib
conda install -y -c conda-forge opencv
conda install -y cython
##################################
for Windows: 
Install Visual Studio 2017
conda install -y pystan

pip install  Box2D
conda install -y  git
pip install git+https://github.com/Kojoley/atari-py.git

conda install -y  swig
pip install Box2D
pip install gym[all]

conda install -y cython
pip install pyglet==1.2.4
pip install gym[box2d]
#################################
pip install pillow ffmpeg gym
pip install gym[all] 
pip install gym[atari]
To install pybulletgym: 
pip install git+https://github.com/benelot/pybullet-gym.git

```
If you want to run the agent with a pretrained model, use the following:
```
python DQNAgent.py --visual True --testPhase True --timeToStart 0 --epsilon 0 
```
If you want to run the agent with the default parameters, then simply run:
```
python DQNAgent.py --resetModel True --resetExp True --timeToStart 100000
```
You may examine the parameters in DQNAgent.py to adjust the model.

