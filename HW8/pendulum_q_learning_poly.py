import gym
import random
import numpy as np
from MDP import TD_poly
from MDP import q_learning_poly

def factorial(n):
    fact = 1
  
    for i in range(1,n+1):
        fact = fact * i

    return fact

def combinations(n, d):
    return factorial(n) / (factorial(n-d) * factorial(d))

def get_state_from_observation(obs):

    cos_theta = obs[0]
    sin_theta = obs[1]
    vel = obs[2]

    theta = np.arcsin(sin_theta)

    if sin_theta > 0 and cos_theta < 0:
        theta = np.pi - theta

    elif sin_theta < 0 and cos_theta < 0:
        theta = -np.pi - theta

    return (theta, vel)

def eps_scheduler(episode: int):
    if episode < 2000: return 0.2
    if episode < 8000: return 0.1
    return 0.02

def lr_scheduler(R_hat: int):
    # if R_hat > -300: return 0.00015
    # # if R_hat > -350: return 0.0015
    # # if R_hat > -600: return 0.025
    # if R_hat > -1000: return 0.00001
    return 0.00001

def build_feature(obs, act):
    x, y, v = obs
    F = np.array([1, x, y, v, act, x**2, y**2, v**2, act**2, x*y, x*v, x*act, y*v, y*act, v*act]).T
    return F

def get_best_Q(W, obs, actions): 
    """
    Takes in weights, observation, and action space,
    returns a tuple: (best action, best action value)
    """
    F = build_feature(obs, actions[1])
    best_q = W@F
    best_a = actions[0]
    for a in actions: 
        F = build_feature(obs, a)
        q = W@F
        if q >= best_q: 
            best_q = q
            best_a = a
    # print(best_q, best_a)
    return best_q, best_a

def eval_policy(env, W, acc, num_episodes=100, num_steps=200):

    R_hat = 0
    for ep in range(num_episodes):
        env.reset()
        R_ep = 0
        observation = env.state
        for _ in range(num_steps):
            _, action = get_best_Q(W, observation, acc)
            observation, reward, _, _, _ = env.step([action])
            observation = get_state_from_observation(observation)
            R_ep += reward
        R_hat += R_ep
        print(f"Reward for ep {ep}: {R_ep}")
    print(f"Average Reward for {num_episodes} episodes: {R_hat/num_episodes}")

if __name__ == '__main__':

    np.random.seed(1)

    env = gym.make('Pendulum-v1')

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    num_states = len(env.observation_space.low)
    # num_states = 2
    print(num_states)
    # Action is one-dimensional (but may take multiple values as before)
    act_dim = 1

    poly_power = 2

    num_features = int(combinations(num_states + act_dim + poly_power, poly_power))
    
    gamma = 0.9
    steps = 200
    num_actions = 8
    goal = -400

    acc = np.linspace(act_min, act_max, num_actions)

    # initialize polynomial approximator
    W = np.zeros(num_features)

    ep = 0
    R_total = 0
    R_hat = -3000   

    # loop until goal is met
    while(1):
        ep+=1
        env.reset()
        #get obs
        observation = env.state
        observation = (np.cos(observation[0]), np.sin(observation[0]), observation[1])
        R = 0
        for _ in range(steps):
            # epsilon greedy pick best action
            epsilon = eps_scheduler(ep)

            # determine the max q-value and action by looping through discrete action space
            if random.random()>epsilon:
                curr_q_value, action = get_best_Q(W, observation, acc)
            else: 
                action = acc[random.randint(0, len(acc)-1)]
                F = build_feature(observation, action)
                curr_q_value = W@F
            
            #take a step in this direction
            new_obs, reward, _, _, _ = env.step([action])
            R+=reward

            # get next max q value
            next_max_q, next_a = get_best_Q(W, new_obs, acc)

            # td target
            target = reward + gamma*next_max_q

            # update the weights for next iteration
            alpha = lr_scheduler(R_hat)
            F = build_feature(observation, action)
            W = W + alpha*(target - curr_q_value)*F

            observation = new_obs
            
        R_total += R
        if ep%100 == 0:
            R_hat = R_total/100
            print(W, alpha*(target - curr_q_value)*F)
            print(f"Ep {ep} Average reward over 100 runs: {R_hat}")
            if R_hat > goal: break
            else: R_total = 0

    test_env = gym.make("Pendulum-v1").env.unwrapped
    eval_policy(test_env, W, acc)


    """
    PSEUDOCODE:
    continuous loop until goal reward is satisfied
      reset the env for this episode
      take an observation, construct the feature vector
      for each step in a set range
          take a new observation, construct t+1 feature vector
          increment the weights by the gradient 
          w' = w - alpha*[2*(R_t + gamma*max_a(W.T@f(S_t+1, a)) - W.T@f(S_t, A_t)) * f(S_t, A_t)]
          above is essentially a gradient descent using the previous observations to bootstrap the current target
          last term f(S_t, A_t) is the features that are responsible for the change
    """
    
