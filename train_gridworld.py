import numpy as np
from gridworld import GridWorld
import random
from matplotlib import pyplot as plt

env = GridWorld(hard_version=False)

env.reset()

def policy_iteration(verbose = True):
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
            for s in range(num_states):
                v = 0
                for act,action_prob in enumerate(pi[s]):
                    for s_new in range(num_states):
                        v += env.p(s_new,s,act)*action_prob*(env.r(s,act) + gamma*V[s_new])
                delta = max(delta, np.abs(v-V[s]))
                V[s] = v
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
            if verbose == True:
                print("Policy Iteration")
                print("------------------------------------------------------")
                print(f"Value function = {V}")
                print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
                print("------------------------------------------------------")
            return V, pi, log


def value_iteration(verbose = True):
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
        for s in range(num_states):
            v = V[s]
            V_temp = np.zeros(num_actions)
            for a in range(num_actions):
                for s_new in range(num_states):
                    V_temp[a] +=  env.p(s_new,s,a)*(env.r(s,a) + gamma*V[s_new])
            V[s] = np.max(V_temp)
            pi[s] = np.eye(num_actions)[np.argmax(V_temp)]
            delta = max(delta, np.abs(v-V[s]))
        log['V'].append(np.mean(V))
        log['iters'].append(iters)
    if verbose == True:
        print("Value Iteration")
        print("------------------------------------------------------")
        print(f"Value Function = {V}")
        print(f"Optimal Policy = {np.argmax(pi,axis=1)}")
        print("------------------------------------------------------")
    return V,pi,log

def epsilon_greedy(Q_state,epsilon):
    if random.uniform(0,1) < epsilon:
        action = random.randint(0,len(Q_state)-1)
    else:
        action = np.argmax(Q_state)
    return action

def TD_0(pi, alpha=0.5, num_episodes = 1000):
    num_states = 25
    V = np.zeros(num_states)
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

def SARSA(alpha=0.5,epsilon=0.1,num_epsiodes=1000,verbose = True):
    num_states = 25
    num_actions = 4
    gamma = 0.95
    Q = np.zeros((num_states,num_actions))
    pi = np.ones((num_states,num_actions))/num_actions
    # V_approx = np.zeros(num_states)
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
        pi[s] = np.eye(num_actions)[np.argmax(Q[s])]
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
        
def Q_Learning(alpha=0.5,epsilon=0.1,num_epsiodes=1000,verbose = True):
    num_states = 25
    num_actions = 4
    gamma = 0.95
    Q = np.zeros((num_states,num_actions))
    pi = np.ones((num_states,num_actions))/num_actions
    log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'V': [],
        'G': [],
        'episodes': [],
        'iters': []
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
        pi[s] = np.eye(num_actions)[np.argmax(Q[s])]
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
    V_policyiter, pi_policyiter, log_policyiter = policy_iteration()
    V_valueiter, pi_valueiter, log_valueiter = value_iteration()
    Q_sarsa,pi_sarsa,log_sarsa = SARSA()
    Q_qlearning,pi_qlearning,log_qlearning = Q_Learning()
    log_list = [log_policyiter,log_valueiter,log_sarsa,log_qlearning]
    pi_list = [pi_policyiter,pi_valueiter,pi_sarsa,pi_qlearning]
    algorithms = {0:"Policy Iteration", 1:"Value Iteration", 2:"SARSA",3:"Q-Learning"}
    for i in range(len(log_list)):
        sp = env.reset()
        log_list[i]['s'].append(sp)
        done = False
        while not done:
            a = np.argmax(pi_list[i][sp])
            (sp, rp, done) = env.step(a)
            log_list[i]['t'].append(log_list[i]['t'][-1] + 1)
            log_list[i]['s'].append(sp)
            log_list[i]['a'].append(a)
            log_list[i]['r'].append(rp)
        
        plt.figure()
        plt.plot(log_list[i]['t'], log_list[i]['s'])
        plt.plot(log_list[i]['t'][:-1], log_list[i]['a'])
        plt.plot(log_list[i]['t'][:-1], log_list[i]['r'])
        plt.title(f"State, Action and Reward for {algorithms[i]}")
        plt.xlabel("Time")
        plt.legend(['s', 'a', 'r'])

        plt.figure()
        if i < 2:
            plt.plot(log_list[i]['iters'],log_list[i]['V'])
            plt.xlabel("Number of Iterations")
            plt.ylabel("Mean of Value Function")
            plt.title(f"Learning curve for number of iterations: {algorithms[i]}")

        else:
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
                plt.title(f"Learning curve for different alpha: {algorithms[i+2]}")
            plt.legend()

            epsilon_vals = np.linspace(0,0.5,11)
            plt.figure()
            for epsilon in epsilon_vals:
                Q_eps, pi_eps, log_eps = SARSA(epsilon=epsilon,verbose=False)
                plt.scatter(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
                plt.title(f"Learning curve for different epsilon: {algorithms[i+2]}")
            plt.legend()
        
        else:
            alpha_vals = np.linspace(0,1,11)
            plt.figure()
            for alpha in alpha_vals:
                Q_alpha, pi_alpha, log_alpha = Q_Learning(alpha=alpha,verbose=False)
                plt.scatter(log_alpha['episodes'],log_alpha['G'],label=f'Alpha={alpha}')
                plt.title(f"Learning curve for different alpha: {algorithms[i+2]}")
            plt.legend()

            epsilon_vals = np.linspace(0,0.5,11)
            plt.figure()
            for epsilon in epsilon_vals:
                Q_eps, pi_eps, log_eps = Q_Learning(epsilon=epsilon,verbose=False)
                plt.scatter(log_eps['episodes'],log_eps['G'],label=f'Epsilon={epsilon}')
                plt.title(f"Learning curve for different epsilon: {algorithms[i+2]}")
            plt.legend()

    plt.show()

if __name__ == '__main__':
    main()