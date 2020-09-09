import numpy as np
import networkx as nx
import gym
import hashlib



GAMMA = 1
# import gym env variables
env = gym.make()
env.seed(1)
ACTION_SPACE = env.action_space.n
OBS_SPACE = env.observation_space
graph = nx.DiGraph()



###############################
# perform in 'incomplete knowledge'
# environment.
# Perform 'randomized q iteration'
# I.e, explore space or go with
# optimal value 
###############################

def flip_coin():
  flip = np.random.random()
  if flip < 0.5:
    return True
  else:
    return False

def mark_exit_node(cur_state):
  cur_hashed_state = hash_state(cur_state)
  exit_node = graph.nodes[cur_hashed_state]
  if exit_node.get('done') is None:
    exit_node['done'] = True
  return 0

def get_done_prop(state):
  cur_hashed_state = hash_state(cur_state)
  exit_node = graph.nodes[cur_hashed_state]
  if exit_node.get('done') is None:
    return None
  else:
    return True

# next_state, next_reward, done
def max_q(cur_state):
  cur_hashed_state = hash_state(cur_state)
  children = graph.successors(cur_hashed_state)
  max_value = -1
  max_child_hash = -1
  for child in children:
    if graph.nodes[child]['value'] > max_value:
      max_value = graph.nodes[child]['value']
      max_child_hash = child 
  # found maximum q. obtain reward to return to user
  max_reward = graph.nodes[max_child_hash]['value']
  # grab done property, if it exists on a node
  done = get_done_prop(max_child_hash)
  return max_child_hash, max_reward, done


def execute_episode():
  # initialize at one, given that we begin our algorithm by taking an initial step
  # to being our TD(0) flow
  k = 1
  # decide on first action to take.
  # for now, we are choosing a random action from the action space
  action = np.random.randint(0, ACTION_SPACE)
  # create an initial state to begin TD(0) flow
  cur_state, cur_reward, done, info = env.step(action)
  # check to see if a terminal state has been reached
  if done:
    return -1
  # begin policy evaluation loop
  while True:
    k += 1
    # flip a coin
    flipped_coin = flip_coin()
    ###
    # Choose to explore space or choose optimal value
    ###
    if flipped_coin:
      # choose optimal value
      next_state, next_reward, done = max_q(cur_state)
      # choose next optimal node
    else:
      # decide on next action to take (explore space).
      # for now, we are choosing a random action from the action space.
      action = np.random.randint(0, ACTION_SPACE)
      # take step in the env
      next_state, next_reward, done, info = env.step(action)
      # update value function for given node
      update_value_function(cur_state, next_state, cur_reward, next_reward, k)
    # check to see if we have reached a terminal state
    if done:
      # mark node as an exit node
      mark_exit_node(cur_state)
      # reset our env
      env.reset()
      break
    # continue to next state in policy
    cur_state = next_state
    cur_reward = next_reward
  return 0

def connect_nodes(cur_hashed_state, next_hashed_state):
  # check to see if edge already exists
  if not graph.has_edge(cur_hashed_state, next_hashed_state):
    graph.add_edge(cur_hashed_state, next_hashed_state)

def hash_state(state):
   state_as_string = "".join(str(e) for e in state)
   state_as_bytes = str.encode(state_as_string)
   hashed_state = hashlib.sha256(state_as_bytes).hexdigest()
   return hashed_state

def get_value_function(hashed_state):
  # randomly initialize value function for given state if the value has not
  # been initialized already
  if not graph.has_node(hashed_state):
    value = np.random.random()
    graph.add_node(hashed_state, value=value)
    return value
  else:
    value = graph.nodes[hashed_state]['value']
    return value


# update value definition for given node
def update_value_function(cur_state, next_state, cur_reward, next_reward, k):
  cur_hashed_state = hash_state(cur_state)
  next_hashed_state = hash_state(next_state)
  cur_value = get_value_function(cur_hashed_state)
  next_value = get_value_function(next_hashed_state)
  new_value = cur_value + (1/k)*(cur_reward + GAMMA*next_value - cur_value)
  # update value function for current state
  graph.nodes[cur_hashed_state]["value"] = new_value
  # connect nodes in graph (i.e, our MDP) as we cotinue walks through our
  # environment with our defined policy
  connect_nodes(cur_hashed_state, next_hashed_state)

def evaluate(n_episodes):
  # initialize our env
  env.reset()
  # run episodes
  for i in range(n_episodes):
    execute_episode()

evaluate(1)