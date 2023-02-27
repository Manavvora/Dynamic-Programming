import numpy as np
from discrete_pendulum import Pendulum
import random
from matplotlib import pyplot as plt

env = Pendulum()

def epsilon_greedy(Q_state,epsilon):
    if random.uniform(0,1) < epsilon:
        action = random.randint(0,len(Q_state)-1)
    else:
        action = np.argmax(Q_state)
    return action

def SARSA(alpha=0.5,epsilon=0.1,num_epsiodes=10000):
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
        'iters': []
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
    return Q, pi, log

def Q_Learning(alpha=0.5,epsilon=0.1,num_epsiodes=1000):
    gamma = 0.95
    Q = np.zeros((env.num_states,env.num_actions))
    pi = np.ones((env.num_states,env.num_actions))/env.num_actions
    for episode in range(num_epsiodes):
        s = env.reset()
        log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'episodes': []
    }
        done = False
        while not done:
            a = epsilon_greedy(Q[s],epsilon)
            (s_new,r,done) = env.step(a)
            Q[s,a] += alpha*(r + gamma*np.max(Q[s_new]) - Q[s,a])
            s = s_new
        pi[s] = np.eye(env.num_actions)[np.argmax(Q[s])]
        log['V'].append(np.mean(np.max(Q,axis=1)))
        log['episodes'].append(episode)
    return Q, pi, log

def TD_0(pi, alpha=0.5, num_episodes = 1000):
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

def main():
    Q1,pi1,log1 = SARSA()
    print(Q1)
    print(pi1)
    sp = env.reset()
    log1['s'].append(sp)
    done = False
    while not done:
        a = np.argmax(pi1[sp])
        (sp, rp, done) = env.step(a)
        log1['t'].append(log1['t'][-1] + 1)
        log1['s'].append(sp)
        log1['a'].append(a)
        log1['r'].append(rp)
    plt.plot(log1['episodes'],log1['G'])
    plt.show()

if __name__ == '__main__':
    main()