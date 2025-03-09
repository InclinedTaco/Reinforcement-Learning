import numpy as np

states = [0, 1, 2, 3]
actions = [0, 1]


transition_prob = {
    0: {0: [(1.0, 0, 0)], 1: [(1.0, 1, 0)]},
    1: {0: [(1.0, 0, 0)], 1: [(1.0, 2, 0)]},
    2: {0: [(1.0, 1, 0)], 1: [(1.0, 3, 0)]},
    3: {0: [(1.0, 2, 0)], 1: [(1.0, 3, 1)]},
}

gamma = 0.9
theta = 1e-6  
policy=np.zeros((len(states),len(actions)))/len(actions)

V=np.zeros(len(states))

def policy_evaluation(policy,V):
    while True:
        delta=0
        for s in states:
            v=0
            for a in actions:
                for prob,next_state,reward in transition_prob[s][a]:
                    v+=policy[s][a]*prob*(reward+gamma*V[next_state])
            delta=max(delta,abs(v-V[s]))
            V[s]=v
        if delta<theta:
            break
    return V

def softmax(x):
    exp_x=np.exp(x-np.max(x))
    return exp_x/np.sum(exp_x)

def policy_improvement(V):
    policy_stable=True
    for s in states:
        old_action=np.copy(policy[s])
        q_values=np.zeros(len(actions))
        for a in actions:
            for prob,next_state,reward in transition_prob[s][a]:
                q_values[a]+=prob*(reward+gamma*V[next_state])
        policy[s]=softmax(q_values)
        if not np.array_equal(old_action,policy[s]):
            policy_stable=False
    
    return policy,policy_stable

def policy_iteration():
    global policy, V
    while True:
        V = policy_evaluation(policy, V)
        policy, policy_stable = policy_improvement(V)
        if policy_stable:
            break
    return policy, V

optimal_policy, optimal_value = policy_iteration()
print("Optimal Stochastic Policy:\n", optimal_policy)
print("Optimal Value Function:\n", optimal_value)
