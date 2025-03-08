import numpy as np

actions=[0,1]

states=[0,1,2,3]

transition_prob={
   0: {0:[(1.0,0,0)], 1:[(1.0,1,0)]}, # (prob, next_state, reward)
   1: {0:[(1.0,0,0)], 1:[(1.0,2,0)]},
   2: {0:[(1.0,1,0)], 1:[(1.0,3,0)]},
   3: {0:[(1.0,2,0)], 1:[(1.0,3,1)]}
   }

gamma=0.25
theta=0.0001

V=np.zeros(len(states))

def value_iteration():
    while True:
        delta=0
        for s in states:
            old_value=V[s]
            V[s]=max([sum([prob*(reward+gamma*V[next_state]) for prob, next_state, reward in transition_prob[s][a]]) for a in actions])
            delta=max(delta, abs(old_value-V[s]))

        if delta<theta:
            break
          
    policy=np.zeros(len(states),dtype=int)

    for s in states:
        action_vales=[sum(prob*(reward+gamma*V[next_state]) for prob, next_state, reward in transition_prob[s][a]) for a in actions]
        policy[s]=actions[np.argmax(action_vales)]

    return policy, V

optimal_policy, optimal_value = value_iteration()
print("Optimal Policy:", optimal_policy)
print("Optimal Value Function:", optimal_value)






