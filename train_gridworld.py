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

def main():    
    V1, pi1, log = policy_iteration()
    print(V1)
    print("------------------------")
    print(pi1)
    print(np.argmax(pi1,axis=1))
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
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    # plt.plot(log['iters'], log['V'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.show()

if __name__ == '__main__':
    main()