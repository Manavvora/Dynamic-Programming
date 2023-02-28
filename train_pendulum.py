import numpy as np
from discrete_pendulum import Pendulum
import random
from matplotlib import pyplot as plt

env = Pendulum(n_theta=15, n_thetadot=21)

def epsilon_greedy(Q_state,epsilon):
    if random.uniform(0,1) < epsilon:
        action = random.randint(0,len(Q_state)-1)
    else:
        action = np.argmax(Q_state)
    return action

def TD_0(pi, alpha=0.5, num_episodes = 100):
    V = np.zeros(env.num_states)
    gamma = 0.95
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'iters': []
    }
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(pi[s])
            (s_new,r,done) = env.step(a)
            V[s] += alpha*(r + gamma*V[s_new]-V[s])
            s = s_new
    return V


def SARSA(alpha=0.5,epsilon=0.1,num_epsiodes=100,verbose=True):
    gamma = 0.95
    Q = np.zeros((env.num_states,env.num_actions))
    pi = np.ones((env.num_states,env.num_actions))/env.num_actions
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'G': [],
        'episodes': [],
        'iters': [],
        'theta': [],
        'thetadot': []
    }
    for episode in range(num_epsiodes):
        s = env.reset()
        a = epsilon_greedy(Q[s],epsilon)
        done = False
        G = 0
        iters = 0
        while not done:
            (s_new,r,done) = env.step(a)
            iters += 1
            G += r*gamma**(iters-1)
            a_new = epsilon_greedy(Q[s_new],epsilon)
            Q[s,a] += alpha*(r + gamma*Q[s_new,a_new] - Q[s,a])
            s = s_new
            a = a_new
        pi[s] = np.eye(env.num_actions)[np.argmax(Q[s])]
        log['G'].append(G)
        log['episodes'].append(episode)
    V_approx = TD_0(pi)
    if verbose == True:
        print(f"SARSA (Episodes = {num_epsiodes})")
        print("------------------------------------------------------")
        print(f"Q function = {Q}")
        print(f"Value function = {np.max(Q,axis=1)}")
        print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
        print(f"Approximate Value function using TD(0) = {V_approx}")
        print("------------------------------------------------------")
    return Q, pi, log
    return Q, pi, log

def Q_Learning(alpha=0.5,epsilon=0.1,num_epsiodes=100,verbose=True):
    gamma = 0.95
    Q = np.zeros((env.num_states,env.num_actions))
    pi = np.ones((env.num_states,env.num_actions))/env.num_actions
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'G': [],
        'episodes': [],
        'theta': [],
        'thetadot': []
    }
    for episode in range(num_epsiodes):
        s = env.reset()
        done = False
        G = 0
        iters = 0
        while not done:
            a = epsilon_greedy(Q[s],epsilon)
            (s_new,r,done) = env.step(a)
            iters += 1
            G += r*gamma**(iters-1)
            Q[s,a] += alpha*(r + gamma*np.max(Q[s_new]) - Q[s,a])
            s = s_new
        pi[s] = np.eye(env.num_actions)[np.argmax(Q[s])]
        log['G'].append(G)
        log['episodes'].append(episode)
    V_approx = TD_0(pi)
    if verbose == True:
        print(f"Q Learning (Episodes = {num_epsiodes})")
        print("------------------------------------------------------")
        print(f"Q function = {Q}")
        print(f"Value function = {np.max(Q,axis=1)}")
        print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
        print(f"Approximate Value function using TD(0) = {V_approx}")
        print("------------------------------------------------------")
    return Q, pi, log


def main():
    Q_sarsa,pi_sarsa,log_sarsa = SARSA()
    Q_qlearning,pi_qlearning,log_qlearning = Q_Learning()
    log_list = [log_sarsa,log_qlearning]
    pi_list = [pi_sarsa,pi_qlearning]
    algorithms = {0:"SARSA",1:"Q-Learning"}
    for i in range(len(log_list)):
        sp = env.reset()
        log_list[i]['s'].append(sp)
        log_list[i]['theta'].append(env.x[0])
        log_list[i]['thetadot'].append(env.x[1])
        done = False
        while not done:
            a = np.argmax(pi_list[i][sp])
            (sp, rp, done) = env.step(a)
            log_list[i]['t'].append(log_list[i]['t'][-1] + 1)
            log_list[i]['s'].append(sp)
            log_list[i]['a'].append(a)
            log_list[i]['r'].append(rp)
            log_list[i]['theta'].append(env.x[0])
            log_list[i]['thetadot'].append(env.x[1])
        plt.figure()
        plt.plot(log_list[i]['t'], log_list[i]['s'])
        plt.plot(log_list[i]['t'][:-1], log_list[i]['a'])
        plt.plot(log_list[i]['t'][:-1], log_list[i]['r'])
        plt.title(f"State, Action and Reward vs Time for {algorithms[i]}")
        plt.xlabel("Time")
        plt.legend(['s', 'a', 'r'])

        plt.figure()
        plt.plot(log_list[i]['t'], log_list[i]['theta'])
        plt.plot(log_list[i]['t'], log_list[i]['thetadot'])
        plt.title(f"Theta, Theta_dot vs Time for {algorithms[i]}")
        plt.xlabel("Time")
        plt.legend(['theta','theta_dot'])

        plt.figure()
        plt.plot(log_list[i]['episodes'],log_list[i]['G'])
        plt.xlabel("Number of episodes")
        plt.ylabel("Total return (G)")
        plt.title(f"Learning curve for number of episodes: {algorithms[i]}")
    
    for i in range(2):
        if i == 0:
            alpha_vals = np.linspace(0,1,11)
            plt.figure()
            for alpha in alpha_vals:
                Q_alpha, pi_alpha, log_alpha = SARSA(alpha=alpha,verbose=False)
                plt.scatter(log_alpha['episodes'],log_alpha['G'],label=f'Alpha={alpha}')
                plt.title(f"Learning curve for different alpha: {algorithms[i]}")
            plt.legend()

            epsilon_vals = np.linspace(0,0.5,11)
            plt.figure()
            for epsilon in epsilon_vals:
                Q_eps, pi_eps, log_eps = SARSA(epsilon=epsilon,verbose=False)
                plt.scatter(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
                plt.title(f"Learning curve for different epsilon: {algorithms[i]}")
            plt.legend()
        
        else:
            alpha_vals = np.linspace(0,1,11)
            plt.figure()
            for alpha in alpha_vals:
                Q_alpha, pi_alpha, log_alpha = Q_Learning(alpha=alpha,verbose=False)
                plt.scatter(log_alpha['episodes'],log_alpha['G'],label=f'Alpha={alpha}')
                plt.title(f"Learning curve for different alpha: {algorithms[i]}")
            plt.legend()

            epsilon_vals = np.linspace(0,0.5,11)
            plt.figure()
            for epsilon in epsilon_vals:
                Q_eps, pi_eps, log_eps = Q_Learning(epsilon=epsilon,verbose=False)
                plt.scatter(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
                plt.title(f"Learning curve for different epsilon: {algorithms[i]}")
            plt.legend()

if __name__ == '__main__':
    main()