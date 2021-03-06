import tensorflow as tf
from collections import deque
import numpy as np
from Debugger import Debugger
e,d = exec, Debugger()
e(d.gIS())
d.sDS(True)
class Replaytypes:
    NormalReplay = 1 # combined experience replay
    CER = 2 # Combined experience replay
class DQN:
    def __init__(self,
                session,
                img_size = (120,120),
                n_channel = 3, 
                numObservations = 5,
                optimizer = 'adam',
                lr = 1e-4,
                checkPointPath = 'checkpoints/',
                loadOldModel = True,
                activation = 'relu',
                numActions = 10,
                gamma = .95,
                maxExp = 200000,
                scopeQ = 'Q',
                scopeTarget = 'Target-Q',
                cUpdate = 10000,
                resetModel = False,
                saveTime = 10000,
                timeToStart = 2000,
                checkpointName = None,
                ):
        self.obs_size = img_size + (numObservations,)
        self.checkpointName = checkpointName
        self.timeToStart = timeToStart
        self.saveTime = saveTime
        self.resetModel = resetModel
        self.cUpdate = cUpdate
        self.session = session
        self.gamma = gamma
        self.maxExp = maxExp
        self.experience = deque()        
        self.checkPointPath = checkPointPath
        self.n_channel = n_channel
        self.optimizer = optimizer
        self.lr = lr
        self.numActions = numActions
        self.step = 0
        self.act = None
        self.opt = None
        self.activation = activation
        self.setActivation()
        self.scopeQ = scopeQ
        self.scopeTarget = scopeTarget
        self.setOptimizer()
        self.wInit = tf.keras.initializers.VarianceScaling(scale = 1, distribution = 'truncated_normal')
        self.obs = tf.compat.v1.placeholder(tf.float32, shape = (None,) + self.obs_size, name = 'obs-img')
        self.targObs = tf.compat.v1.placeholder(tf.float32,shape = (None,) + self.obs_size, name = 'targ-obs-img')
        self.y = tf.compat.v1.placeholder(tf.float32, shape = (None), name = 'y')
        self.targy = tf.compat.v1.placeholder(tf.float32, shape = (None), name = 'targy')
        self.bInit = tf.keras.initializers.Zeros()
        self.targQValue = self.build_DQN(self.targObs,scopeTarget)
        self.QValue = self.build_DQN(self.obs, scopeQ)
        self.saver = tf.compat.v1.train.Saver()
        self.setLoss()
        self.session.run(tf.compat.v1.initialize_all_variables())
        self.loadCheckPoint()

    def loadCheckPoint(self):
        """
        loads checkpoint
        """
        
        checkpoint = tf.compat.v1.train.get_checkpoint_state("checkpoints")
        
        success = True
        e(g('checkpoint.model_checkpoint_path'))
        if checkpoint and checkpoint.model_checkpoint_path and not self.resetModel: 
            try:
                if self.checkpointName is not None: 
                    checkPath = self.checkpointName
                else: checkPath = checkpoint.model_checkpoint_path
                e(g('checkPath'))
                self.saver.restore(self.session,checkPath)
                print("weights restored")
                
            except: 
                success = False
                print("path not valid")
            finally: 
                if not success: print("Model Not Loaded")
                if success: print("Model Loaded")
                
        else: print("Could not find Weights")
    def setLoss(self):
        """
        sets loss function, optimizer 
        """
        self.actionChosen = tf.compat.v1.placeholder("float", (None,) + (self.numActions,))
        Q = tf.compat.v1.reduce_sum(tf.multiply(self.QValue,self.actionChosen), reduction_indices =1)
        self.loss = tf.reduce_mean(tf.square(self.y - Q)) 
        self.train_op = self.opt.minimize(loss = self.loss, ) 
    def setActivation(self):
        """
        sets activation funciton to be used for all layers of neural network
        """
        if self.activation == 'relu': self.act = tf.nn.relu
        elif self.activation == 'leaky_relu': self.act = tf.nn.leaky_relu
        elif self.activation == 'elu': self.act = tf.nn.elu
    def setOptimizer(self):
        """
        Sets optimizer
        """
        if self.optimizer == 'adam': 
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate = .00025,beta1 = .9, beta2 = .999, epsilon = 1e-8,)
        elif self.optimizer == 'sgd':
            self.opt = tf.keras.optimizeres.SGD
        else:
            print("Not supported optimizer, usin   Adam:")
            self.opt = tf.keras.optimizers.Adam
    def conv2D(self,x , filters, stride, kernel_size, name):
        """
        Params: 
            x - input 
            filters - number of filters for output 
            stride - strides for moving arround input image
            kernel_size - size of kernel_size
            name - name of layer, for use in updating weights of target
        returns: conv2d layer using above parameters
        """
        
        return tf.compat.v1.layers.conv2d(inputs = x,
            filters = filters,
            kernel_size = kernel_size,
            strides = stride ,padding = 'valid',
            name = name )
 
    def build_DQN(self, x, scopeName):
        """
        Params:
            x - input 
            scopeName - name to be used for model
        returns - tensorflow graph under scopeName 
        """
        conv2D = self.conv2D
        with tf.compat.v1.variable_scope(scopeName): 
            x = x / 255
            
            convLayer1 = self.act(conv2D(x, filters = 32, stride = [4,4], kernel_size = 8, name = 'convLayer1')) 
            e(g('convLayer1.shape'))
            maxPoolLayer = tf.nn.max_pool(convLayer1, ksize = [2,2], strides = [1,2,2,1],padding = 'SAME', name = 'maxPoolLayer')
            e(g('maxPoolLayer.shape'))
            convLayer2 = self.act(conv2D(convLayer1,filters = 64, stride = [2,2], kernel_size = 4,name = 'convLayer2'))
            e(g('convLayer2.shape'))
            convLayer3 = self.act(conv2D(convLayer2, filters = 64, stride = [1,1], kernel_size = 3, name = 'convLayer3'))
            e(g('convLayer3.shape'))
            flatL3 = tf.compat.v1.layers.flatten(convLayer3, name = 'flatten')
            e(g('flatL3.shape'))
            dense1 = self.act(tf.compat.v1.layers.dense(flatL3,512, name = 'dense1'))
            e(g('dense1.shape'))
            QValue = tf.compat.v1.layers.dense(dense1,self.numActions, name = 'QValue')
            return QValue
    def argmax(self, state):
        """
        Params: 
            state - 84 by 84 by 4 state
        returns: action following optimal policy
        """
        state = np.reshape(state, (1,) + state.shape)
        qvalue = self.QValue.eval(session = self.session,feed_dict = {self.obs: state})[0]

        return np.argmax(qvalue)
    def getParamsFromScope(self,scope):
        """
        Params: 
            scope - string name of scope
        returns: sorted list of all parameters under scope
        """
        params = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith(scope)]
        return sorted(params, key=lambda v: v.name)
    def assignParams(self,params1, params2):
        """
        Params: 
            params2 - list of sorted parameters to have their values changed
            params1 - list of sorted parameters to be assigned
        returns: list of functions which can be ran using session.run
        """
        fList = [] 
        for v1, v2 in zip(params1, params2):
            op = v2.assign(v1)
            fList += [op]
        return fList

    def updateTargetNetwork(self, timeStep):
        """
        Params: 
            timeStep - current global step in environment
        output: updates parameters every self.cUpdate steps
        """
        
        if timeStep % self.cUpdate == 0:
            print("updating target network")
            Q_params = self.getParamsFromScope(self.scopeQ)
            Targ_params = self.getParamsFromScope(self.scopeTarget)
            fList = self.assignParams(Q_params,Targ_params)
            self.session.run(fList)

    def save(self,timeStep):
        """
        Params: 
            timeStep - current global step in environment
        output: saves model to path self.checkpoints
        """
        if timeStep % self.saveTime == 0:
            print("model saved")
            self.saver.save(self.session,self.checkPointPath + 'saved_dqn', global_step = timeStep)
    def getListsFromTupleList(self,TupleList):
        """
        Params: 
            TupleList - list of tuples of arbitrary length
        returns - returns lists where the number of lists returns
        depends on the length of each tuple 
        """
        Lists = ()
        for i in range(len(TupleList[0])):
            myList = [element[i] for element in TupleList]
            Lists += (myList,)
        
        return Lists        
    def getBatch(self, batch, recentExp, expType):
        """
        Params: 
            batch - length of batch 
            recentExp - recent experience for use in combined experience replay
            expType - type of replay buffer, choices are in enum Replaytypes
        returns: lists  where number of lists is len(recentExp) from  batch 
        and length of each list is len(batch)
        
        """
        
        if expType == Replaytypes.NormalReplay: 
            return self.getListsFromTupleList(batch)
        elif expType == Replaytypes.CER:
            batch[-1] = recentExp
            return self.getListsFromTupleList(batch)
    def trainOnExperience(self, batch, timeStep,recentExp, expType = 1):
        """
        Params: 
            batch - length of batch
            timeStep - current global step in environment
            recentExp - recent experience to be used in CER
            expType - type of replay buffer
        output: runs model with batch as input.
        """
        if timeStep > self.timeToStart:
            states,actions,rewards,nextStates,terminals = self.getBatch(batch,recentExp,expType)

            QvalueTargs = self.targQValue.eval(feed_dict = {self.targObs : nextStates})
            yTarg = [rewards[i] if terminals[i] else rewards[i] + self.gamma * np.max(QvalueTargs[i]) for i in range(len(batch))]
            self.train_op.run(feed_dict = {self.y : yTarg,self.actionChosen : actions,self.obs : states,})
            self.save(timeStep)
            self.updateTargetNetwork(timeStep)
                
        

