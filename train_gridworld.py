import numpy as np
from gridworld import GridWorld
import random
from matplotlib import pyplot as plt

env = GridWorld(hard_version=False)

env.reset()

def policy_iteration():
    num_states = 25
    num_actions = 4
    V = np.zeros(num_states)
    pi = np.ones([num_states,num_actions])/num_actions
    # pi_new = np.zeros(num_states)
    theta = 1e-6
    gamma = 0.95
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'iters': []
    }
    while True:
        iters = 0
        delta = np.inf
        while delta > theta:
            # V = np.zeros(num_states)
            delta = 0
            iters += 1
            print("New iteration")
            for s in range(num_states):
                v = 0
                for act,action_prob in enumerate(pi[s]):
                    for s_new in range(num_states):
                        v += env.p(s_new,s,act)*action_prob*(env.r(s,act) + gamma*V[s_new])
                delta = max(delta, np.abs(v-V[s]))
                V[s] = v
                print(delta)
            print('------------')
        log['V'].append(np.mean(V))
        log['iters'].append(iters)
        policy_stable = True
        for s in range(num_states):
            chosen_a = np.argmax(pi[s])
            pi_new = np.zeros(num_actions)
            for a in range(num_actions):
                for s_new in range(num_states):
                    pi_new[a] += env.p(s_new,s,a)*(env.r(s,a) + gamma*V[s_new])
            best_a = np.argmax(pi_new)
            if chosen_a != best_a:
                policy_stable = False
            pi[s] = np.eye(num_actions)[best_a]
        if policy_stable:
            return V, pi, log


def value_iteration():
    num_states = 25
    num_actions = 4
    V = np.zeros(num_states)
    V_temp = np.zeros(num_actions)
    pi = np.ones([num_states,num_actions])/num_actions
    # pi_new = np.zeros(num_states)
    theta = 1e-6
    gamma = 0.95
    delta = 1
    iters = 0
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'iters': []
    }
    while delta > theta:
        delta = 0
        iters += 1
        print("New iteration")
        for s in range(num_states):
            v = V[s]
            V_temp = np.zeros(num_actions)
            for a in range(num_actions):
                for s_new in range(num_states):
                    V_temp[a] +=  env.p(s_new,s,a)*(env.r(s,a) + gamma*V[s_new])
            V[s] = np.max(V_temp)
            pi[s] = np.eye(num_actions)[np.argmax(V_temp)]
            delta = max(delta, np.abs(v-V[s]))
            print(delta)
        print('------------')
        log['V'].append(np.mean(V))
        log['iters'].append(iters)
    return V,pi,log

def epsilon_greedy(Q_state,epsilon):
    if random.uniform(0,1) < epsilon:
        action = random.randint(0,len(Q_state)-1)
    else:
        action = np.argmax(Q_state)
    return action

def SARSA(alpha=0.5,epsilon=0.1,num_epsiodes=1000):
    num_states = 25
    num_actions = 4
    gamma = 0.95
    Q = np.zeros((num_states,num_actions))
    pi = np.zeros((num_states,num_actions))
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
        a = epsilon_greedy(Q[s],epsilon)
        done = False
        while not done:
            (s_new,r,done) = env.step(a)
            a_new = epsilon_greedy(Q[s_new],epsilon)
            Q[s,a] += alpha*(r + gamma*Q[s_new,a_new] - Q[s,a])
            s = s_new
            a = a_new
        pi[s] = np.eye(num_actions)[np.argmax(Q[s])]
        log['V'].append(np.mean(np.max(Q,axis=1)))
        log['episodes'].append(episode)
    return Q, pi, log
        
def Q_Learning(alpha=0.5,epsilon=0.1,num_epsiodes=1000):
    num_states = 25
    num_actions = 4
    gamma = 0.95
    Q = np.zeros((num_states,num_actions))
    pi = np.zeros((num_states,num_actions))
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
        pi[s] = np.eye(num_actions)[np.argmax(Q[s])]
        log['V'].append(np.mean(np.max(Q,axis=1)))
        log['episodes'].append(episode)
    return Q, pi, log

def TD_0(pi, alpha, num_episodes = 1000):
    num_states = 25
    V = np.zeros(num_states)
    gamma = 0.95
    done = False
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
        while not done:
            a = np.argmax(pi[s])
            (s_new,r,done) = env.step(a)
            V[s] += alpha*(r + gamma*V[s_new]-V[s])
            s = s_new
    return V

def main():    
    #V1, pi1, log = policy_iteration()
    # V1, pi1, log = value_iteration()
    Q1,pi1,log = SARSA(alpha=0.5,epsilon=0.1,num_epsiodes=1000)
    Q2,pi2,lo2 = SARSA(alpha=0.5,epsilon=0.1,num_epsiodes=1000)
    V1 = np.max(Q1,axis=1)
    print(V1)
    print("------------------------")
    V2 = np.max(Q2,axis=1)
    print(V2)
    print(np.argmax(pi1,axis=1))
    print(np.argmax(pi2,axis=1))
    sp = env.reset()
    log['s'].append(sp)
    done = False
    while not done:
        a = np.argmax(pi1[sp])
        (sp, rp, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(sp)
        log['a'].append(a)
        log['r'].append(rp)

    # # Plot data and save to png file
    # plt.plot(log['t'], log['s'])
    # plt.plot(log['t'][:-1], log['a'])
    #plt.plot(log['episodes'], log['V'])
    # plt.plot(log['t'][:-1], log['r'])
    # plt.legend(['s', 'a', 'r'])
    #plt.show()

if __name__ == '__main__':
    main()