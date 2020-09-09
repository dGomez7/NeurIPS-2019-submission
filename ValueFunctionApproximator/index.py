import numpy as np


bias = 0.1
step_size = 0.2
NUM_STATES = env.state_space.n
NUM_ACTIONS = env.action_space.n


class ValueFuncApprox:
    def _init__(self, alpha=0.1):
        self.weights = np.zeros(NUM_STATES)
        self.alpha = alpha
    def value(self, state):
        v = np.dot(self.weights, state) + bias
        return v
    def learn(self, target, prediction): 
        difference = step_size*(target - prediction)*self.weights + bias
        weights += difference
        

def take_action(self, action):
    state, reward, done, info = env.step(action)
    return state, reward, done, info


# monte-carlo method
action = np.random.random()
state, reward, done, info = take_action(action)
if done:
    break
sum += reward
prediction = value(state)
learn(sum, prediction)

# td0 method (one-step look ahead)
action = np.random.random()
state, reward, done, info = env.step(action)
# same concept, but we will work with the state prior
# to serve as input into our 'learning' method
if done:
    break
sum += reward
prediction = value(prior_state)
learn(sum, prediction)
prior_state = state