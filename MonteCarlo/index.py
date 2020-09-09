import numpy as np


NUM_STATES = 19
START = 9
END_0 = 0
END_1 = 20


class ValueFunction:
    def __init__(self, alpha=0.1):
        self.weights = np.zeros(NUM_STATES + 2)
        self.alpha = alpha
    
    def value(self, state):
        v = self.weights[state]
        return v
    
    def learn(self, state, delta):
        self.weights[state] += self.alpha * delta

class RandomWalk:
    def __init__(self, start=START, end=False, lmbda=0.4, debug=False):
        self.actions = ["left", "right"]
        self.state = start # current state
        self.end = end
        self.lmbda = lmbda
        self.states = []
        self.reward = 0
        self.debug = debug
        self.rate_truncate = 1e-3

    def choose_action(self):
        action = np.random.choice(self.actions)
        return action
    
    def take_action(self, action):
        new_state = self.state
        if not self.end:
            if action == "left":
                new_state = self.state - 1
            # action is right
            else:
                new_state = self.state + 1
            if new_state in [END_0, END_1]:
                self.end = True
        self.state = new_state
        return self.state
    
    def give_reward(self, state):
        if state == END_0:
            return -1
        elif state == END_1:
            return 1
        else:
            return 0

def play(self, value_func, rounds=100):
    for _ in range(rounds):
        self.reset()
        action = self.choose_action()
        self.states = [self.state]
        while not self.end:
            state = self.take_action(action) # next state
            self.reward = self.give_reward(state) # next state-reward
            self.states.append(state)
            action = self.choose_action()
        if self.debug:
            print("total states {} end at {} reward {}".format(len(self.states), self.state, self.reward))
        # End of game. Do a forward update
        T = len(self.states) - 1
        for t in range(T):
            # start from time t
            state = self.states[t]
            gtlambda = 0
            for n in range(1, T - t):
                # compute G_t:t+n
                gttn = self.gt2tn(value_func, t, t+n)
                lambda_power = np.power(self.lmbda, n - 1)
                gtlambda += lambda_power * gttn
                if lambda_power < self.rate_truncate:
                    break
            gtlambda *= 1 - self.lmbda
            if lambda_power >= self.rate_truncate:
                gtlambda += lambda_power * self.reward
            delta = gtlambda - value_func.value(state)
            value_func.learn(state, delta)

def gt2tn(self, value_func, start, end):
    # only the last reward is non-zero
    reward = self.reward if end == len(self.states) - 1 else 0
    state = self.states[end]
    res = reward + value_func.value(state)
    return res