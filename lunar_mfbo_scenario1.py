
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


import numpy as np
from gym import spaces
def compute_trajHf(max_steps,**kwargs):
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
    trajHf = [ob]
    for _ in range(max_steps):
        iter_time += 1
        po, _ = modelddpg.predict(obs, deterministic=True)
        action=po
        ob, reward, done, _ = env.step(action)
        trajHf.append(ob)
        r+= reward 
        done = done or iter_time >= max_steps
        if done:
            break
        return trajHf, {'reward':r}

def sutH(max_steps,x0):
    state_per = np.zeros(2)
    state_per[0:2] += x0[0:2]
    vel_per = np.zeros(2)
    vel_per[0:2] += x0[2:4]
    return compute_trajHf(max_steps, state_per=state_per,vel_per=vel_per)

def compute_trajLf(max_steps,**kwargs):
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
    trajLf = [ob]
    for _ in range(max_steps):
        iter_time += 1
        action=po
        ob, reward, done, _ = env.step(action)
        noise_x = np.random.normal(0,0.02,1)
        noise_v_x=np.random.normal(0,0.01,1)
        ob[0]+=noise_x
        ob[2]+=noise_v_x
        trajLf.append(ob)
        r+= reward 
        done = done or iter_time >= max_steps
        if done:
            break
        return trajLf, {'reward':r}

def sutL(max_steps,x0):
    state_per = np.zeros(2)
    state_per[0:2] += x0[0:2]
    vel_per = np.zeros(2)
    vel_per[0:2] += x0[2:4]
    return compute_trajLf(max_steps, state_per=state_per,vel_per=vel_per)

rand_nums1 = [210029653,14586213,258741945,12345678]
rand_nums2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,14,21,86,74,12]
rand_nums3=[91, 30, 32, 34, 36, 38,54,87,46,27,15]
rand_nums4=[101, 113, 134, 156, 194, 202, 213, 111, 129, 200,205,316,389,158,249]
rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 1714,8742,9851,3218,1694,3794]
rand_nums6=[1461, 8194, 6927, 5075, 4903, 3799, 6268, 8155, 5502, 1187,2147,1587,3578,6584,2893]
rand_nums7=[64846, 28856, 43210, 70661, 14700, 21044, 58191, 17243, 24958, 80194,52147,15984,12365,52478,35871]
rand_nums8=[54239, 69118, 51184, 57468, 57945, 78075, 34142, 78062, 33150,
            64048,25874,14796,32571,15874,36598]
rand_nums9=[ 64639, 81307, 16913,12587,14956,35896,12584,96358]
rand_nums10=[684823,145896]
rand_num=[rand_nums1,rand_nums2 ,rand_nums3 ,rand_nums4,rand_nums5,rand_nums6 ,rand_nums7 ,rand_nums8,rand_nums9 ,rand_nums10]  #

######## - - - - - --- - -   Utils ----- ---  - - - - ##############

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
           ContinuousParameter('vy', 0,2),InformationSourceParameter(2)])

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

# Assign costs
from emukit.core.acquisition import Acquisition
# Define cost of different fidelities as acquisition function
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]
    
    @property
    def has_gradients(self):
        return True
    
    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)

from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
low_fidelity_cost = 1
high_fidelity_cost = 5


global x_array,y_array
class test_module:
    global y_array
    def __init__(self,sutl,suth, bounds, spec=None,f_tree=None,
                 normalizer=False,seed=None, **kwargs):

        self.system_under_test_L=sutl
        self.system_under_test_H=suth
        self.f_tree=f_tree
        self.bounds = bounds
        self.normalizer=normalizer
        self.seed=seed


        if 'cost_model' in kwargs:
            self.cost_model = kwargs['cost_model']
        else:
            self.cost_model = lambda x: 1
  
        if 'init_sample' in kwargs:
            self.init_sample = kwargs['init_sample']
        else:
            self.init_sample = 2*len(bounds)


        if 'with_ns' in kwargs:
            self.with_ns = kwargs['with_ns']
        else:
            self.with_ns = False


        if 'exp_weight' in kwargs:
            self.k = kwargs['exp_weight']
        else:
            self.k = 10

        # Optimize retsrats for hyper parameter optimization for GPs
        if 'optimize_restarts' in kwargs:
            self.optimize_restarts = kwargs['optimize_restarts']
        else:
            self.optimize_restarts = 1


        if 'XL' in kwargs:
            self.XL = kwargs['XL']
        else:
            self.XL = []

        if 'XH' in kwargs:
            self.XH = kwargs['XH']
        else:
            self.XH = []
         
    def initialize(self):
        global low_exp_num
        global high_exp_num
        global real_low_ce
        global real_high_ce
        global valid_low_ce
        global valid_high_ce
        real_high_ce=0

        real_low_ce=0
        valid_low_ce=0
        valid_high_ce=0
        global X_ns
        global XL, XH,YL, YH  
        if len(self.XL) == 0:
            XL = sample_from(self.init_sample, self.bounds)
            self.XL = XL

        if len(self.XH) == 0:
            o=self.init_sample//2
            XH = np.atleast_2d(np.random.permutation(XL)[:o])
            self.XH = XH
        # print(XH)
        global trajsL,trajsH
        global XL_ns, XH_ns,YL, YH
        trajsL = []
        trajsH = []
        for x in self.XL:
            trajsL.append(self.system_under_test_L(x))
        self.f_acqu=self.f_tree[0]
        YL = self.f_acqu.eval_robustness(trajsL)

        for x in self.XH:
            trajsH.append(self.system_under_test_H(x))
        self.f_acqu=self.f_tree[1]
        YH = self.f_acqu.eval_robustness(trajsH)
        low_exp_num=self.init_sample
        high_exp_num=self.init_sample//2

        for x in self.XL:
               trajL=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_xlow=self.f_acqu.eval_robustness(trajL)
               self.f_acqu=self.f_tree[1]
               trajH=self.system_under_test_H(x)
               f_xhigh=self.f_acqu.eval_robustness(trajH)
               if (f_xlow<0) and (f_xhigh<0):
                 real_low_ce=1 + real_low_ce

        for x in self.XH:
               self.f_acqu=self.f_tree[1]
               traj_H=self.system_under_test_H(x)
               f_x_high=self.f_acqu.eval_robustness(traj_H)
               traj_L=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_x_low=self.f_acqu.eval_robustness(traj_L)
               if (f_x_low>0) and (f_x_high<0):
                 real_high_ce=1 + real_high_ce
        global x_array,y_array
        x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
       
    global XL, XH,YL, YH
    global y_array

    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            global XL, XH,YL, YH
            global low_exp_num
            global high_exp_num
            global real_low_ce
            global real_high_ce
            global valid_low_ce
            global valid_high_ce
            print('BO iteration:', ib)
            global x_array,y_array
            kern_low = GPy.kern.RBF(4,ARD=True)
            #kern_low.lengthscale.constrain_bounded(0.01, 0.5)
            kern_err = GPy.kern.RBF(4,ARD=True)
            #kern_err.lengthscale.constrain_bounded(0.01, 0.5)
            multi_fidelity_kernel = LinearMultiFidelityKernel([kern_low, kern_err])
            gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, 2,None)
            gpy_model.mixed_noise.Gaussian_noise.fix(0.00001)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(0.00001)
            GPmodel = GPyMultiOutputWrapper(gpy_model, 2, 1, verbose_optimization=True)
            GPmodel.optimize()
            cost_acquisition = Cost([low_fidelity_cost, high_fidelity_cost])
            acquisition = MultiInformationSourceEntropySearch(GPmodel, bound) / cost_acquisition
            acquisition_optimizer=MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(bound), bound)
            new_x,val_acq=acquisition_optimizer.optimize(acquisition)
            #print(new_x)
            
            if new_x[0][-1]==0.:
               x=new_x[0][0:4]  
               XL=np.vstack((XL, x))
               low_exp_num=1+low_exp_num
               trajsL=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_xl=self.f_acqu.eval_robustness(trajsL)
               self.f_acqu=self.f_tree[1]
               trajsH=self.system_under_test_H(x)
               f_test_ce=self.f_acqu.eval_robustness(trajsH)
               if (f_xl<0) and (f_test_ce<0):
                 print("It's a valid counterexample")
                 valid_low_ce=1+valid_low_ce

               print(f"f_xl= {f_xl}")
               YL=np.vstack((YL, f_xl))    
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
            else:
               a=new_x[0][0:4]
               XH=np.vstack((XH, a))
               high_exp_num =1+high_exp_num
               trajsH=self.system_under_test_H(a)
               self.f_acqu=self.f_tree[1]
               f_xh=self.f_acqu.eval_robustness(trajsH)
               if f_xh<0:
                  valid_high_ce=1+valid_high_ce
               print(f"f_xh= {f_xh}")
               YH=np.vstack((YH, f_xh))
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
        global n
        n=0
        global sume_real_h_ce
        sume_real_h_ce=0
        global init_ce_lf 
        global sum_real_ce
        global MF_c
        MF_c=0
        sum_real_ce=0
        init_ce_lf=0
        global min_val
        min_val = y_array.min()
        for i in range(len(y_array)):
          if y_array[i][0]<0:
            n=n+1
            #print('counterexample found')
        print(n)
        sume_real_h_ce=(valid_high_ce)+(real_high_ce)
        sum_real_ce=(valid_high_ce)+(valid_low_ce)+(real_high_ce)+(real_low_ce)
        MF_c=5*(high_exp_num)+1*(low_exp_num)

# Safety specification
from numpy import mean
import math
from math import pi
import warnings
warnings.filterwarnings('ignore')
nums=[]
h_num=[]
l_num=[]
min_phi=[]
initial_Hf_ce=[]
initial_Lf_ce=[]
MFBO_cost=[]
real_num_ce=[]
ce_h=[]
############### specifications for lf
def pred2(trajLf):
    traj = trajLf[0]
    for state in traj:
        last_state = state[0]
    return 0.1-np.abs(last_state)

def pred3(trajLf):
      traj = trajLf[0]
      for state in traj:
         theta=state[4]
      return (pi/4)-np.abs(theta)

def pred4(trajLf):
      traj =trajLf[0]
      for state in traj:
        d=state[5]
      return 0.2-np.abs(d)
####################Specifications for hf

def pred5(trajHf):
    traj = trajHf[0]
    for state in traj:
        last_state = state[0]
    return 0.1-np.abs(last_state)

def pred6(trajHf):
      traj = trajHf[0]
      for state in traj:
         theta=state[4]
      return (pi/4)-np.abs(theta)

def pred7(trajHf):
      traj = trajHf[0]
      for state in traj:
        d=state[5]
      return 0.2-np.abs(d)

######################################################

global n
for r in rand_nums1:
      np.random.seed(r)
      node0 = pred_node(f=pred4)
      node1 = pred_node(f=pred3)
      node2 = pred_node(f=pred2)
      node4_lf = min_node(children=[node1, node2,node0])
      node0_h = pred_node(f=pred5)
      node1_h = pred_node(f=pred6)
      node2_h = pred_node(f=pred7)
      node4_hf = min_node(children=[node0_h,node1_h,node2_h])
      node=[node4_lf,node4_hf]
      TM_ns = test_module(bounds=bounds,suth=lambda x0: sutH(600,x0),sutl=lambda x0: sutL(600,x0),
                          f_tree = node,init_sample =7, with_ns=True, exp_weight=2, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(20)
      nums.append(n)
      h_num.append(high_exp_num)
      l_num.append(low_exp_num)
      min_phi.append(min_val)
      initial_Hf_ce.append(real_high_ce)
      initial_Lf_ce.append(real_low_ce)
      MFBO_cost.append(MF_c)
      real_num_ce.append(sum_real_ce)
      ce_h.append(sume_real_h_ce)
      print(f"number of experiments in HF simulator : {h_num}")
      print(f"number of experiments in LF simulator : {l_num}")
      print(f"min of phi after 5 BO iterations: {min_phi}")
      print(f"{nums} counterexample found")
      print(f"mean of counterexampe over every exp: {mean(nums)}")
      print(f" Cost is {MFBO_cost}")
      print(f"mean of cost over every exp: {mean(MFBO_cost)}")
      print(f"number of counterexamples found in HF simulator: {ce_h}")
      print(f" number of valid counterexamples are : {real_num_ce}")
