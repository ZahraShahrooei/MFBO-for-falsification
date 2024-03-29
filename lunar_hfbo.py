
#dependencies
!pip install stable_baselines3 
import stable_baselines3
import gym
!pip install gpy
import GPy
from stable_baselines3 import PPO
import gym
import numpy as np
!pip3 install box2d-py
from stable_baselines3 import DDPG
!pip install emukit
import emukit
# !pip uninstall scipy
# !pip install scipy==1.4.1

env = gym.make('LunarLanderContinuous-v2')
model = DDPG("MlpPolicy", env, action_noise=None, verbose=1)
model.learn(total_timesteps=40000, log_interval=10)
model.save("ddpg_LunarLanderContinuous-v2")
# env = model.get_env()
modelddpg = DDPG.load("ddpg_LunarLanderContinuous-v2")
obs = env.reset()


#Trajectories
# SUT
def compute_traj(max_steps,**kwargs):

    env.reset()
    if 'init_state' in kwargs:
        env.env.lander.position=kwargs['init_state']
    if 'init_velocity' in kwargs:
        env.env.lander.linearVelocity = kwargs['init_velocity']
    # State perturbation
    if 'state_per' in kwargs:
        state_per = kwargs['state_per']
    # Velocity perturbation
    if 'vel_per' in kwargs:
        vel_per = kwargs['vel_per']
    # env.env.lander.
    env.env.lander.position[0] = env.env.lander.position[0] + state_per[0]
    env.env.lander.position[1] = env.env.lander.position[1] + state_per[1]
    env.lander.linearVelocity[0]=env.lander.linearVelocity[0]+vel_per[0]
    env.lander.linearVelocity[1]=env.lander.linearVelocity[1]+vel_per[1]
    ob=np.array([env.env.lander.position[0],env.env.lander.position[1],
                     env.lander.linearVelocity[0],env.lander.linearVelocity[1],env.env.lander.angle,
                     env.env.lander.angularVelocity,0,0])  
    iter_time = 0
    r = 0
    done=False
    traj = [ob]
    for _ in range(max_steps):
        iter_time += 1
        po, _ = modelddpg.predict(obs, deterministic=True)
        action=po
        ob, reward, done, _ = env.step(action)
        traj.append(ob)
        r+= reward 
        done = done or iter_time >= max_steps
        if done:
            break
        return traj, {'reward':r}

def sut(max_steps,x0):
    state_per = np.zeros(2)
    state_per[0:2] += x0[0:2]
    vel_per = np.zeros(2)
    vel_per[0:2] += x0[2:4]
    return compute_traj(max_steps, state_per=state_per,vel_per=vel_per)

######## Utils ##############
import numpy as np
def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]

    return np.array(sampled_lb).T

bounds = [(-0.5, 0.5)] # Bounds on the X perturbations
bounds.append((0, 3)) # Bounds on the Y perturbations
bounds.append((-2, 2)) ## Bounds on the x velocity perturbations
bounds.append((0, 2)) # Bounds on the y velocity perturbations
from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter
bound = ParameterSpace([ContinuousParameter('p_x', -0.5,0.5),
           ContinuousParameter('p_y', 0,3),
           ContinuousParameter('vx', -2,2), 
           ContinuousParameter('vy', 0,2)])

#Function tree ##
import numpy as np
import GPy
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

rand_nums1 = [3248332584,1304673443,3857496775,2668478815,278535713,1762150547,
              788841329,2525132954,677754898,754758634,5684574,214789653,14586213,258741945,12345678]
rand_nums2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,14,21,86,74,12]
rand_nums3=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38,54,87,46,27,15]
rand_nums4=[101, 113, 134, 156, 194, 202, 213, 111, 129, 200,205,316,389,158,249]
rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 1714,8742,9851,3218,1694,3794]
rand_nums6=[1461, 8194, 6927, 5075, 4903, 3799, 6268, 8155, 5502, 1187,2147,1587,3578,6584,2893]
rand_nums7=[64846, 28856, 43210, 70661, 14700, 21044, 58191, 17243, 24958, 80194,52147,15984,12365,52478,35871]
rand_nums8=[54239, 69118, 51184, 57468, 57945, 78075, 34142, 78062, 33150,
            64048,25874,14796,32571,15874,36598]
rand_nums9=[63951, 36835, 59249, 17176, 32123, 54118, 79720, 64639, 81307, 16913,12587,14956,35896,12584,96358]
rand_nums10=[347957, 510020, 545416, 613511, 673274, 619204, 630790, 627544,
       127016, 390172,123658,145698,625893,965823,145896]
rand_num=[rand_nums1 ,rand_nums2 ,rand_nums3 ,rand_nums4 ,rand_nums5,rand_nums6 ,rand_nums7,rand_nums8,rand_nums9 ,rand_nums10] 
ns_Failure_count=[]
ns_details_r3 = []
num_nonsmooth=[]

from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
from emukit.model_wrappers import GPyModelWrapper
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
from sklearn.decomposition import KernelPCA
from emukit.core.optimization.optimizer import Optimizer
from emukit.core.optimization.optimizer import OptLbfgs
import copy
import GPy
class test_module:
    def __init__(self, sut, bounds, spec=None,f_tree=None,optimizer=None,
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

        if 'with_ns' in kwargs:
            self.with_ns = kwargs['with_ns']
        else:
            self.with_ns = False
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
        # Sending in pre sampled data
        if 'X' in kwargs:
            self.X = kwargs['X']
        else:
            self.X = []
    def initialize(self):
        global num_ce
        num_ce=0
        global X_ns
        if len(self.X) == 0:
            X = sample_from(self.init_sample, self.bounds)
            self.X = X
        trajs = []
        for x in self.X:
            trajs.append(self.system_under_test(x))
        Y = self.f_acqu.eval_robustness(trajs)
        for i in range(len(Y)):
          if Y[i][0]<0:
            num_ce+=1
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
            global Hf_model
            Hf_model = GPyModelWrapper(self.ns_GP)

    global ns_min_val
    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            print('BO iteration:', ib)
            global X_ns
            global num_ce
            if self.with_ns:
              global Hf_model
              Hf_acq=EntropySearch(Hf_model, bound)
              x,_ = self.optimizer.optimize(Hf_acq,None)
            #-----------------------------------------------------
              trajs = [self.system_under_test(x_i) for x_i in x]
              f_x = self.f_acqu.eval_robustness(trajs)
              if f_x <0:
                #print("counterexample found")
                num_ce=num_ce + 1
              #print(f_x)
              self.ns_X = np.vstack((self.ns_X, x))
              X_ns = self.ns_X
              Y=np.vstack((self.ns_GP.Y, f_x))
              self.ns_GP.set_XY(X_ns,Y)
              self.ns_GP.optimize_restarts(self.optimize_restarts)
              Hf_model = GPyModelWrapper(self.ns_GP)
        global ns_min_val
        if self.with_ns:
          ns_min_val = self.ns_GP.Y.min()
          print(num_ce)

# Safety specification
from emukit.core.optimization import GradientAcquisitionOptimizer
nums=[]
num_ce_real=[]
import math
from math import pi
import warnings
p=[]
warnings.filterwarnings('ignore')
# 2. The lander should not go beyond the flag -0.4<=x_pos<=0.4
def pred2(traj):
    reward = traj[1]['reward']
    traj = traj[0]
    for state in traj:
        last_state = state[0]
    return 0.1-np.abs(last_state)
def pred3(traj):
      traj = traj[0]
      for state in traj:
         theta=state[4]
      return (pi/4)-np.abs(theta)
def pred4(traj):
      traj = traj[0]
      for state in traj:
        d=state[5]
      return 0.2-np.abs(d)
#######################################################
for r in rand_nums5:
      np.random.seed(r)
      node0 = pred_node(f=pred4)
      node1 = pred_node(f=pred3)
      node2 = pred_node(f=pred2)
      node4 = min_node(children=[node1, node2,node0])
      TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(600,x0), optimizer=GradientAcquisitionOptimizer(bound),f_tree = node4,init_sample =4, with_ns=True,
                          optimize_restarts=1, exp_weight=2, normalizer=False)
      TM_ns.initialize()
      TM_ns.run_BO(9) 
      num_ce_real.append(num_ce)
      print(num_ce_real)
      print(np.mean(num_ce_real))
      p.append(ns_min_val)
      print(f"min of phi is: {p}")
