
!pip install stable_baselines3 
import stable_baselines3
import gym
!pip install gpy
import GPy
from stable_baselines3 import PPO
# Train MountainCar
env = gym.make("MountainCarContinuous-v0")
modelppo = PPO("MlpPolicy", env, verbose=1)
# Train the model
modelppo.learn(total_timesteps=1000000)
# Save the model
modelppo.save("PPO_MC")
# Load the trained model
modelppo = PPO.load("PPO_MC")
# Start a new episode
ob = env.reset()
# What action to take in state `obs`?


#Trajectories
# SUT
# Cost function

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
    print(ob)
    print(np.shape(ob))
    done=False
    traj = [ob]
    while done==False:
        iter_time += 1
        pi, _ = modelppo.predict(ob, deterministic=True)
        action=pi
        ob, reward, done, _ = env.step(action)
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
bounds.append((-0.003, 0.003)) # Bounds on the velocity
bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
bounds.append((0.0005, 0.0025)) # Bounds on the power magnitude

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
        # print(f"trajs in eval: {trajs}")
        Y = np.array([self.f(traj) for traj in trajs])
        # print(Y)
        return Y.reshape(len(Y), 1)


    def find_GP_func(self):
        return self.GP.Y

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
       
        if 'init_sample' in kwargs:
            self.init_sample = kwargs['init_sample']
        else:
            self.init_sample = 2*len(bounds)

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

        # Sending in pre sampled data
        if 'X' in kwargs:
            self.X = kwargs['X']
        else:
            self.X = []


    def initialize(self):
        if len(self.X) == 0:
            X = sample_from(self.init_sample, self.bounds)
            self.X = X

        trajs = []
        for x in self.X:
            trajs.append(self.system_under_test(x))
            print(trajs)
            print(np.shape(trajs))
        Y = self.f_acqu.eval_robustness(trajs)
        print(f" Y is {Y}")

        if self.with_random:
            self.random_X = copy.deepcopy(self.X)
            self.random_Y = Y

    def run_BO(self, iters_BO):
        if self.with_random:
            if self.seed is not None:
                np.random.seed(self.seed)
                sample_from(self.init_sample, self.bounds)
            rand_x = sample_from(iters_BO, self.bounds)
            trajs = []
            for x in rand_x:
                trajs.append(self.system_under_test(x))
            self.random_X = np.vstack((self.random_X, rand_x))
            rand_y = self.f_acqu.eval_robustness(trajs)
            self.random_Y = np.vstack((self.random_Y, rand_y))

        if self.with_random:
             global n
        n=0
        for i in range(len(self.random_Y)):
          if self.random_Y[i][0]<0:
            n+=1
            #print('counterexample found')
        print(n)
        global rand_min_val
        rand_min_val = self.random_Y.min()

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

rand_num=[rand_nums1 ,rand_nums2 ,rand_nums3 ,rand_nums4 ,rand_nums5,rand_nums6 ,rand_nums7 ,rand_nums8 ,rand_nums9 ,rand_nums10]

# Safety specification in paper:
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap
nums=[]
m=[]
p=[]
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
    return min(0.02 - np.abs(v_s))


##########################################


# for m in range(10):
for r in rand_nums1:
      np.random.seed(r)
      node0 = pred_node(f=pred1)
      node1 = pred_node(f=pred2)
      node2 = pred_node(f=pred3)
      node3 = min_node(children=[node0, node2])
      node4 = max_node(children=[node3, node1])
      TM = test_module(bounds=bounds, sut=lambda x0: sut(x0, max_steps=350),
                      f_tree=node4, init_sample=9, with_smooth=False,
                      with_random=True,
                      optimize_restarts=1, exp_weight=10,
                      normalizer=True)
      TM.initialize()
      TM.run_BO(16)
      nums.append(n)
      p.append(rand_min_val)
      print(nums)
      print(p)
