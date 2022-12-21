import gym
import random
import numpy as np
from MDP import TD
from MDP import q_learning

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

def get_state(pos, vel, observation):
    """
    approximate bin indexes for observation from continuous space to MDP state
    """    
    p, v = observation
    p_bucket = np.digitize(p, pos) - 1 
    v_bucket = np.digitize(v, vel) - 1 
    return (p_bucket,v_bucket)

def eps_scheduler(episode: int):
    if episode < 150: return 0.1
    return 0.01

def lr_scheduler(R_hat: int):
    if R_hat > -300: return 0.0001
    if R_hat > -350: return 0.0015
    if R_hat > -600: return 0.025
    if R_hat > -1000: return 0.15
    return 0.5

def eval_policy(env, policy, the, vel, acc, num_steps=200, num_episodes=100):
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
            t_idx, v_idx = get_state(the, vel, observation)
            action = acc[policy[t_idx][v_idx]]
            observation, reward, terminated, _, _ = env.step([action])
            observation = get_state_from_observation(observation)
            R_ep += reward
            if terminated: break
        R_hat += R_ep
        print(f"Reward for ep {ep}: {R_ep}")
    print(f"Average Reward for {num_episodes} episodes: {R_hat/num_episodes}")

if __name__ == '__main__':

    np.random.seed(1)

    env = gym.make('Pendulum-v1')

    theta_min = -np.pi
    theta_max = np.pi

    vel_min = env.observation_space.low[2]
    vel_max = env.observation_space.high[2]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    gamma = 0.9
    num_data_per_state = 24
    num_actions = 8
    steps = 200
    goal = -265
    
    the = np.linspace(theta_min, theta_max, num_data_per_state)
    vel = np.linspace(vel_min, vel_max, num_data_per_state)
    acc = np.linspace(act_min, act_max, num_actions)

    #initialize Q table
    Q = np.zeros(shape=(len(the), len(vel), len(acc)))

    ep = 0
    R_total = 0
    R_hat = -3000
    while(1): 
        ep += 1
        env.reset()
        #get observation
        observation = env.state
        terminated = False
        R = 0
        for step in range(steps):
            #change observation to discrete state index
            t_idx, v_idx = get_state(the, vel, observation)

            #epsilon greedy
            epsilon = eps_scheduler(ep)
            if random.random()>epsilon:
                action = acc[np.argmax(Q[t_idx][v_idx])]
            else: 
                action = acc[random.randint(0, len(acc)-1)]

            a_idx = np.digitize(action, acc) - 1
            #take a step
            new_obs, reward, terminated, _, _ = env.step([action])
            R+=reward

            # Q learning algorithm equation 
            curr_q_value = Q[t_idx][v_idx][a_idx]

            new_obs = get_state_from_observation(new_obs)
            new_t_idx, new_v_idx = get_state(the, vel, new_obs)
            
            #getting the next max value
            next_max_q = np.amax(Q[new_t_idx][new_v_idx])

            #td target
            target = reward + gamma*next_max_q

            alpha = lr_scheduler(R_hat)

            # if its the last step, just add by the one setp reward
            if step == steps-1: Q[t_idx][v_idx][a_idx] = curr_q_value + alpha*(reward)
            else: Q[t_idx][v_idx][a_idx] = curr_q_value + alpha*(target - curr_q_value)

            observation = new_obs
        # print(f"Training Ep {ep}, Reward: {R}")
        R_total += R
        if ep%100 == 0:
            R_hat = R_total/100
            print(f"Ep {ep} Average reward over 100 runs: {R_hat}")
            if R_hat > goal: break
            else: R_total = 0
    
    test_env = gym.make("Pendulum-v1").env.unwrapped
    policy = np.argmax(Q, axis = 2)
    eval_policy(test_env, policy, the, vel, acc)

