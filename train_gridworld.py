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
   
    pi = np.zeros(num_states)
    # pi_new = np.zeros(num_states)
    theta = 1e-6
    gamma = 0.95
    #delta = 1
    while True:
        delta = np.inf
        while delta > theta:
            delta = 0
            for s in range(num_states):
                V_new = np.zeros(num_states)
                for s_new in range(num_states):
                    V_new[s] += env.p(s_new,s,pi[s])*(env.r(s,pi[s]) + gamma*V[s_new])
                delta = max(delta, np.abs(V[s]-V_new[s]))
                print(delta)
            V = V_new
        policy_stable = True
        for s in range(num_states):
            chosen_a = pi[s]
            pi_new = np.zeros(num_actions)
            for a in range(num_actions):
                for s_new in range(num_states):
                    pi_new[a] += env.p(s_new,s,a)*(env.r(s,a) + gamma*V[s_new])
                # if maximum < one_step:
                #     maximum = one_step
                #     best_a = a
            pi[s] = np.argmax(pi_new)
            if chosen_a != pi[s]:
                policy_stable = False
        if policy_stable:
            return V, pi
            
V1, pi1 = policy_iteration()
