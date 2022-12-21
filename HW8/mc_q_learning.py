import gym
import random
import numpy as np
from MDP import TD
from MDP import q_learning


def eval_policy(env, policy, pos, vel, acc, num_steps=1000, num_episodes=100):
    """
    Inputs: ENV
            Policy: a 2d array (pos, vel) where the index is that state and the value is the index of the action
            pos: position space to determine discrete state from observation
            vel: velocity space to determine discrete state from observation
            acc: action space to map index to action value
    Outputs: Average reward
    """
    R_hat = 0
    for ep in range(num_episodes):
        env.reset()
        R_ep = 0
        observation = env.state        
        for step in range(num_steps):
            p_idx, v_idx = get_state(pos, vel, observation)
            action = acc[policy[p_idx][v_idx]]
            observation, reward, terminated, _, _ = env.step([action])
            R_ep += reward
            if terminated: break
        R_hat += R_ep
        print(f"Reward for ep {ep}: {R_ep}")
    print(f"Average Reward for {num_episodes} episodes: {R_hat/num_episodes}")

def eps_scheduler(episode: int):
    if episode < 10: return 0.9
    if episode < 100: return 0.5
    else: return 0.1

def get_state(pos, vel, observation):
    """
    approximate bin indexes for observation from continuous space to MDP state
    """    
    p, v = observation
    p_bucket = np.digitize(p, pos) - 1 
    v_bucket = np.digitize(v, vel) - 1 
    return (p_bucket,v_bucket)

def get_policy(Q_table):
    """
    inputs: a Q table (numpy array with shape N x N x A) where N is the number of 
            states for position and velocity and A is the number of actions
    returns: a 2d array (pos x vel) where the element corresponding to each state is the INDEX of the optimal action
    """
    rv = np.argmax(Q_table, axis = 2)
    return rv

if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0').env.unwrapped

    np.random.seed(1)

    pos_min = env.observation_space.low[0]
    pos_max = env.observation_space.high[0]

    vel_min = env.observation_space.low[1]
    vel_max = env.observation_space.high[1]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    gamma = 1

    # Hyper-parameters
    num_data_per_state = 15
    num_actions = 6
    episodes = 1000
    epsilon = 0.1
    alpha = 0.01

    #discretize space
    pos = np.linspace(pos_min, pos_max, num_data_per_state)
    vel = np.linspace(vel_min, vel_max, num_data_per_state)
    acc = np.linspace(act_min, act_max, num_actions)
    
    # initialize arbitrary Q table
    Q = np.zeros(shape=(len(pos), len(vel), len(acc)))
    
    # iterate for a certain number of eps
    for ep in range(episodes):
        env.reset()
        observation = env.state
        terminated = False
        R = 0
        while not terminated: 
            #get state from observation
            p_idx, v_idx = get_state(pos, vel, observation)

            # epsilon greedy: if random greater than epsilon, then do greedy action
            epsilon = eps_scheduler(ep)
            if random.random() > epsilon: 
                action = acc[np.argmax(Q[p_idx][v_idx])]
            # do random action
            else: 
                action = acc[random.randint(0, len(acc)-1)]
            
            a_idx = np.digitize(action, acc) - 1 
            #take a step
            new_observation, reward, terminated, truncated, info = env.step([action])
            R += reward
            """
            Q(S, A) = Q(S, A) + a*[R + gamma * max_a( Q(S', a) ) - Q(S, A)]
            """
            curr_q_value = Q[p_idx][v_idx][a_idx]

            new_p_idx, new_v_idx = get_state(pos, vel, new_observation)
            next_max_q = np.amax(Q[new_p_idx][new_v_idx])

            # R + gamma*max_a(Q(S', a))
            # next_max_q is the approximated long term rewards
            td_target_q_value = reward + gamma * next_max_q

            # update the q value at current state, if its the last state, we don't care about future rewards. 
            if terminated: Q[p_idx][v_idx][a_idx] = curr_q_value + alpha*(reward)
            else: Q[p_idx][v_idx][a_idx] = curr_q_value + alpha*(td_target_q_value - curr_q_value)

            # update the state
            observation = new_observation
        print(f"Training Ep {ep}, Reward: {R}")

    test_env = gym.make('MountainCarContinuous-v0').env.unwrapped
    policy = get_policy(Q)
    print(policy)
    eval_policy(test_env, policy, pos, vel, acc)



    

