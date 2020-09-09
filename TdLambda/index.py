import numpy as np
import numpy as np
import networkx as nx
import gym
import hashlib
import keras
from tensorflow import keras
import tensorflow as tf


# import gym env variables
env = gym.make()
env.seed(1)
OBS_SPACE = env.observation_space
NUM_ACTIONS = env.action_space.n
# ACTION_SPACE = env.action_space.n
GAMMA = 1
STATE_DIM = env.state_space.n

class Agent:

    def __init__(self):
        self.value_approximator = create_value_func_approximator()
    
    def create_value_func_approximator():
        value_approximator = keras.models.Sequential()
        value_approximator.add(keras.layers.Dense(64, activation='relu'))
        value_approximator.add(keras.layers.Dense(64, activation='relu'))
        value_approximator.add(keras.layers.Dense(1, activation=None))
        return value_approximator
    # runs a forward pass through the net to 
    def run_forward_pass(state):
        # expand the dimensions of the state, allowing it to be fed into
        # our net
        state = np.expand(state, axis=0)
        # make prediction for value function of given state
        value = self.value_approximator.predict(state)
        return value
    
    def compute_loss(logit, true_value, rewards):
        
        
    ### Agent Memory ###
    class Memory:
        def __init__(self): 
            self.clear()

        # Resets/restarts the memory buffer
        def clear(self): 
            self.observations = []
            self.actions = []
            self.rewards = []

        # Add observations, actions, rewards to memory
        def add_to_memory(self, new_observation, new_action, new_reward): 
            self.observations.append(new_observation)
            # update the list of actions with new action
            self.actions.append(new_action)
            #update the list of rewards with new reward
            self.rewards.append(new_reward)

def hash_state(state):
   state_as_string = "".join(str(e) for e in state)
   state_as_bytes = str.encode(state_as_string)
   hashed_state = hashlib.sha256(state_as_bytes).hexdigest()
   return hashed_state

class ValueFunction:
    def __init__(self, alpha=0.1):
        self.weights = {}
        self.alpha = alpha
    
    def value(self, state):
        if state not in self.weights:
            self.weights[state] = 0
        return self.weights[state]
    
    def learn(self, state, delta):
        self.weights[state] += self.alpha * delta

class RandomWalk:
    def __init__(self, done=False, debug=False, lmbda=0.4):
        self.actions = NUM_ACTIONS

        self.done = done
        self.lmbda = lmbda
        self.states = []
        self.rewards = []
        self.debug = debug
        self.rate_truncate = 1e-3
    
    def reset(self):
        self.states = []
        self.rewards = []
        self.done = False
    
    def choose_action(self):
        action = np.random.choice(self.actions)
        return action
    
    def take_action(self, action):
        state, reward, done, info = env.step(action)
        if done:
            self.done = done
        return state, reward, info
    
    def gt2tn(self, value_func, start, end):
        reward = 0
        # compute the total reward for the given range
        for i in range(start, end + 1):
            reward += self.rewards[i]
        state = self.states[end]
        res = reward + value_func.value(state)
        return res

    def play(self, value_func, rounds=2):
        for _ in range(rounds):
            # instantiate environment and reset our random walk class
            self.reset()
            env.reset()
            # initiate with a random action
            action = self.choose_action()
            while not self.done:
                state, reward, info = self.take_action(action)
                hashed_state = hash_state(state)
                self.rewards.append(reward)
                self.states.append(hashed_state)
                action = self.choose_action()
            if self.debug:
                print("total states {} end at {}".format(len(self.states), self.state))
            # end of game: compute lambda rewards: i.e, a forward update
            T = len(self.states) - 1
            for t in range(T):
                state = self.states[t]
                gtlambda = 0
                for n in range(1, T - t):
                    # compute Gt:t+n
                    gttn = self.gt2tn(value_func, t, t+n)
                    lambda_power = np.power(self.lmbda, n-1)
                    gtlambda += lambda_power * gttn
                    if lambda_power < self.rate_truncate:
                        break
                gtlambda *= 1 - self.lmbda
                if lambda_power >= self.rate_truncate:
                    # add last reward - discounted by lambda - if our 
                    # truncation rate was not reached
                    gtlambda += lambda_power * self.rewards[t]
                delta = gtlambda - value_func.value(state)
                value_func.learn(state, delta)


# start learining
value_func = ValueFunction()
rand_walk = RandomWalk()
rand_walk.play(value_func)