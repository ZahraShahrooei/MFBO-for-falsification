
!pip install stable_baselines3 
import stable_baselines3
import gym
!pip install gpy
import GPy
from stable_baselines3 import PPO
# Train an agent using Soft Actor-Critic on Pendulum-v0
env = gym.make("MountainCarContinuous-v0")
modelppo = PPO("MlpPolicy", env, verbose=1)
# Train the model
modelppo.learn(total_timesteps=20000)
# Save the model
modelppo.save("PPO_MC")
# Load the trained model
modelppo = PPO.load("PPO_MC")
# Start a new episode
ob = env.reset()
# What action to take in state `obs`?


#Trajectories
# SUT
import numpy as np
from gym import spaces
def compute_traj(**kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'goal_pos' in kwargs:
        gp = kwargs['goal_pos']
        env.env.goal_position = gp
    if 'max_speed' in kwargs:
        ms = kwargs['max_speed']
        env.env.max_speed = ms
        env.env.low_state = \
            np.array([env.env.min_position, - env.env.max_speed])
        env.env.high_state = \
            np.array([env.env.max_position, env.env.max_speed])
        env.env.observation_space = \
            spaces.Box(env.env.low_state, env.env.high_state)
    if 'power' in kwargs:
        pow = kwargs['power']
        env.env.power = pow
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf

    iter_time = 0
    r = 0
    done=False
    traj = [ob]
    while done==False:
        iter_time += 1
        #action, _ = pi.act(False, ob)
        pi, _ = modelppo.predict(ob, deterministic=True)
        action=pi
        ob, reward, done, _ = env.step(action)
        # print(ob)
        traj.append(ob)
        r+= reward 
        done = done or iter_time >= max_steps
        if done:
            break
    return traj, {'reward':r, 'iter_time': iter_time}

def sut(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_traj(max_steps=max_steps, init_state=x0[0:2],goal_pos=x0[2], max_speed=x0[3], power=x0[4])

######## Utils ##############

def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]

    return np.array(sampled_lb).T

bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.0001, 0.0001)) # Bounds on the velocity
bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
bounds.append((0.0005, 0.0025)) # Bounds on the power magnitude
!pip install emukit
import emukit 
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
from emukit.model_wrappers import GPyModelWrapper
from emukit.core import ContinuousParameter, ParameterSpace

bound = ParameterSpace([ContinuousParameter('p', -0.6,-0.4), 
           ContinuousParameter('v', -0.025,0.025),
           ContinuousParameter('gp', 0.4,0.6), 
           ContinuousParameter('ms', 0.055,0.075),
           ContinuousParameter('pm', 0.0005, 0.0025)])

#Function tree ##


import numpy as np
#import GPy
import copy

# Class Tree Node!
class tree_node():
    def __init__(self, children, f=None, df=None):
        self.children = children
        self.f = f
        self.df = df

    def evaluate(self, X,  **kwargs):
        self.cn_data = [child.evaluate(X, **kwargs) for child in self.children]
        return self.f(np.array(self.cn_data), axis=0)

    def eval_df(self, X, **kwargs):
        loc = self.df(np.array(self.cn_data), axis=0)
        cn_df_data = [child.eval_df(X, **kwargs) for child in self.children]
        return cn_df_data[loc]

    def init_GPs(self, X, trajs, **kwargs):
        for child in self.children:
            child.init_GPs(X, trajs, **kwargs)

    def update_GPs(self, X, trajs, **kwargs):
        for child in self.children:
            child.update_GPs(X, trajs, **kwargs)

    def eval_robustness(self, trajs):
        cn_data = [child.eval_robustness(trajs) for child in self.children]
        return self.f(np.array(cn_data), axis=0)

    def find_GP_func(self):
        cn_data = [child.find_GP_func() for child in self.children]
        return self.f(np.array(cn_data), axis=0)

# Different types of nodes!
# Max and Min Node
class max_node(tree_node):
    def __init__(self,children, f=np.amax, df=np.argmax):
        super(max_node, self).__init__(children, f,df)

class min_node(tree_node):
    def __init__(self, children, f=np.amin, df=np.argmin):
        super(min_node, self).__init__(children, f,df)

# Predicate Node
class pred_node(tree_node):
    def __init__(self, children=None, f=None):
        super(pred_node, self).__init__(children, f)
        self.Y = []

    def evaluate(self, X, **kwargs):
        X = np.atleast_2d(X)

        # If mode is True evaluate in GP mode

        if 'k' in kwargs:
            k=kwargs['k']
        else:
            k = 10

        m, v = self.GP.predict(X)
        return m - k*np.sqrt(v)

    def eval_df(self, X, **kwargs):
        X = np.atleast_2d(X)

        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 10
        m,v = self.GP.predict(X)
        dm, dv = self.GP.predictive_gradients(X)
        dm = dm[:, :, 0]
        return dm - (k/2)*(dv/np.sqrt(v))

    def init_GPs(self, X, trajs, **kwargs):
        for traj in trajs:
            self.Y.append(self.f(traj))
        self.Y = np.array(self.Y)
        self.Y.resize(len(self.Y),1)
        if 'kernel' in kwargs:
            kernel = kwargs['kernel']
        else:
            kernel = GPy.kern.Matern32(X.shape[1])
        if 'normalizer' in kwargs:
            normalizer=kwargs['normalizer']
        else:
            normalizer=False
        self.GP = GPy.models.GPRegression(X= X, Y=self.Y,
                                          kernel=copy.deepcopy(kernel),
                                          normalizer=normalizer)

        if 'optimize_restarts' in kwargs:
            self.GP.optimize_restarts(kwargs['optimize_restarts'])
        else:
            self.GP.optimize()

    def update_GPs(self, X, trajs, **kwargs):
        ys = []

        trajs = np.atleast_2d(trajs)
        for traj in trajs:
            ys.append(self.f(traj))
        ys = np.array(ys)
        ys.resize(len(ys), 1)
        self.Y = np.vstack((self.Y, ys))

        self.GP.set_XY(X, self.Y)

        if 'optimize_restarts' in kwargs:
            self.GP.optimize_restarts(kwargs['optimize_restarts'])
        else:
            self.GP.optimize()

    def eval_robustness(self, trajs):
        trajs = np.atleast_2d(trajs)
        Y = np.array([self.f(traj) for traj in trajs])
        return Y.reshape(len(Y), 1)


    def find_GP_func(self):
        return self.GP.Y
#utils.sample_from()

'''
This file defines the testing module. This needs the following:
1. The system under test
2. The specification or the function which we are trying to minimize
3. Domains of the uncertainities
'''
from sklearn.decomposition import KernelPCA
import copy
import GPy

class test_module:
    def __init__(self, sut, bounds, spec=None,f_tree=None, optimizer=None,
                 normalizer=False,seed=None, **kwargs):
        self.system_under_test = sut

        # Choosing the optimizer function
        if spec is None:
            self.f_acqu = f_tree
        else:
            self.spec = spec
            

        self.bounds = bounds
        self.normalizer=normalizer
        self.seed=seed

        if 'cost_model' in kwargs:
            self.cost_model = kwargs['cost_model']
        else:
            self.cost_model = lambda x: 1

        # Choosing the optimizers
        if 'opt_name' in kwargs:
            self.optimizer = select_opt(kwargs['opt_name'])(bounds, **kwargs)
        elif optimizer is None:
            self.optimizer = sample_opt(bounds=bounds, cost=self.cost_model)
        else:
            self.optimizer = optimizer

        # Number of samples for initializing GPs
        if 'init_sample' in kwargs:
            self.init_sample = kwargs['init_sample']
        else:
            self.init_sample = 2*len(bounds)

        # Model GPs for the top level requirement, potentially modeling
        # non-smooth function
        if 'with_ns' in kwargs:
            self.with_ns = kwargs['with_ns']
        else:
            self.with_ns = False

        # Random sampling
        if 'with_random' in kwargs:
            self.with_random = kwargs['with_random']
        else:
            self.with_random = False

        # Exploration weight for GP-LCB
        if 'exp_weight' in kwargs:
            self.k = kwargs['exp_weight']
        else:
            self.k = 10

        # Optimize retsrats for hyper parameter optimization for GPs
        if 'optimize_restarts' in kwargs:
            self.optimize_restarts = kwargs['optimize_restarts']
        else:
            self.optimize_restarts = 1


        # Search in lower dimension
        if 'low_dim' in kwargs:
            self.using_kpca=True
            self.low_dim = kwargs['low_dim']
            if 'kernel_type' in kwargs:
                self.kernel = kwargs['kernel_type'](self.low_dim)
        elif 'kernel' in kwargs:
            self.kernel = kwargs['kernel']
            self.using_kpca = True
            self.low_dim = self.kernel.input_dim
        else:
            self.using_kpca=False
            if 'kernel_type' in kwargs:
                self.kernel = kwargs['kernel_type'](len(bounds))
            else:
                self.kernel = GPy.kern.Matern32(len(bounds), ARD=True)

        if self.using_kpca:
            if isinstance(self.optimizer, lbfgs_opt) or \
                    isinstance(self.optimizer, direct_opt):
                print('Can use only sample_opt or delta_opt!')
                print('Changing optimizer to sample_opt!')
                self.optimizer = sample_opt(bounds, **kwargs)

        # Sending in pre sampled data
        if 'X' in kwargs:
            self.X = kwargs['X']
        else:
            self.X = []


    def initialize(self):
        global X_ns
        if len(self.X) == 0:
            X = sample_from(self.init_sample, self.bounds)
            self.X = X

        trajs = []
        for x in self.X:
            trajs.append(self.system_under_test(x))
        Y = self.f_acqu.eval_robustness(trajs)
        
        if self.with_ns:
            self.ns_X = copy.deepcopy(self.X)
            if self.using_kpca:
                self.kpca_ns = KernelPCA(kernel='rbf', fit_inverse_transform=True,
                          copy_X=True, n_components=self.low_dim)
                X_ns = self.kpca_ns.fit_transform(self.ns_X)
            else:
                X_ns = copy.deepcopy(self.ns_X)
            self.ns_GP = GPy.models.GPRegression(X_ns, Y,
                                        kernel=copy.deepcopy(self.kernel),
                                        normalizer=self.normalizer)


    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            print('BO iteration:', ib)
            if self.with_ns:

                Hf_model = GPyModelWrapper(self.ns_GP)
                def f(X):
                  global X_ns

                  Hf_acq=EntropySearch(Hf_model, bound)
                  return -Hf_acq.evaluate(X_ns)

                x,f = self.optimizer.optimize(f=lambda x: f(x))
                # x,f = self.optimizer.optimize(f=lambda x: f(x)[0],
                #                               df = lambda x:f(x)[1])
                trajs = [self.system_under_test(x_i) for x_i in x]
                f_x = self.f_acqu.eval_robustness(trajs)
                self.ns_X = np.vstack((self.ns_X, np.atleast_2d(x)))
                if self.using_kpca:
                    X_ns = self.kpca_ns.fit_transform(self.ns_X)
                else:
                    X_ns = self.ns_X
                self.ns_GP.set_XY(X_ns,
                                  np.vstack((self.ns_GP.Y, np.atleast_2d(f_x))))
                self.ns_GP.optimize_restarts(self.optimize_restarts)

        if self.with_ns:
            self.ns_min_val = self.ns_GP.Y.min()
            self.ns_min_loc = self.ns_GP.Y.argmin()
            self.ns_min_x = self.ns_GP.X[self.ns_min_loc]

            self.ns_count = np.sum(self.ns_GP.Y < 0)
            self.ns_ce = np.flatnonzero(self.ns_GP.Y < 0)

rand_nums1 = [3188388221,1954593344,2154016205,3894811078,3493033583,3248332584,1304673443,3857496775,2668478815,278535713,1762150547,788841329,2525132954,677754898,754758634]
rand_nums2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
rand_nums3=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,75,96]
rand_nums4=[101, 113, 134, 156, 194, 202, 213, 111, 129, 200, 91, 81, 82, 71, 78]
rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 1714, 7053,
       4131, 7929, 7799, 5766]
rand_nums6=[1461, 8194, 6927, 5075, 4903, 3799, 6268, 8155, 5502, 1187, 7833,
       3916, 7906, 3815, 3587]
rand_nums7=[64846, 28856, 43210, 70661, 14700, 21044, 58191, 17243, 24958, 80194,
       65943, 58561, 24073, 68194, 69265]
rand_nums8=[54239, 69118, 51184, 57468, 57945, 78075, 34142, 78062, 33150,
            64048, 65056, 48293, 35515, 50506, 20161]
rand_nums9=[63951, 36835, 59249, 17176, 32123, 54118, 79720, 64639, 81307, 16913, 
       66005, 22091, 78671, 29591, 74848]
rand_nums10=[347957, 510020, 545416, 613511, 673274, 619204, 630790, 627544,
       127016, 390172, 231790, 414417, 561875, 376595, 632379]

rand_num=[rand_nums1 ,rand_nums2 ,rand_nums3 ,rand_nums4 ,rand_nums5 ,rand_nums6 ,rand_nums7 ,rand_nums8 ,rand_nums9 ,rand_nums10]     


ns_Failure_count=[]

ns_details_r3 = []

num_nonsmooth=[]

# Safety specification in paper:
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
def pred1(traj):
    traj = traj[0]
    x_s = np.array(traj).T[0]
    up = min(-0.4 - x_s)
    low = min(x_s + 0.6)
    return min(up,low)

def pred2(traj):
    iters = traj[1]['iter_time']
    return -iters/350.

def pred3(traj):
    traj=traj[0]
    v_s = np.array(traj).T[1]
    return min(0.025 - np.abs(v_s))


##################### ######################


for m in range(10):
  for r in rand_num[m]:
      np.random.seed(r)
      node0_ns = pred_node(f=pred1)
      node1_ns = pred_node(f=pred2)
      node2_ns = pred_node(f=pred3)
      node3_ns = min_node(children=[node0_ns, node2_ns])
      node4_ns = max_node(children=[node3_ns, node1_ns])
      TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350), optimizer=None,
                      f_tree = node4_ns,init_sample = 6, with_ns=True, optimize_restarts=1, exp_weight=2, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(14)
      ns_Failure_count.append(TM_ns.ns_count)
      ns_smooth_vals = np.array(TM_ns.ns_GP.Y)
      ns_details_r3.append([np.sum(TM_ns.ns_GP.Y < 0),
                           TM_ns.ns_min_x,
                          TM_ns.ns_min_val,
                           TM_ns.ns_min_loc])
      num_nonsmooth.append([np.sum(TM_ns.ns_GP.Y <0)])
      print(num_nonsmooth)
      print(ns_Failure_count)
