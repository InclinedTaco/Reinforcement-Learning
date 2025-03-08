import numpy as np

states=[0,1,2,3]

actions=[0,1]

transition_prob={
   0: {0:[(1.0,0,0)], 1:[(1.0,1,0)]}, # (prob, next_state, reward)
   1: {0:[(1.0,0,0)], 1:[(1.0,2,0)]},
   2: {0:[(1.0,1,0)], 1:[(1.0,3,0)]},
   3: {0:[(1.0,2,0)], 1:[(1.0,3,1)]}
   }
   
gamma=0.25 #discount factor

policy=np.zeros(len(states))
V=np.zeros(len(states)) #value function

def policy_evaluation(policy,V,e): #e is the error threshold
   while True:
    delta=0
    for s in states:
      v=0
      for prob,next_state,reward in transition_prob[s][policy[s]]:
        v+=prob*(reward+gamma*V[next_state])
      delta=max(delta,abs(V[s]-v))
      V[s]=v
    
    if delta<e:
       break
    
   return V
    
    
def policy_improvement(policy,V):
    policy_stable=True
    for  s in states:
      old_action=policy[s]
      action_value=[]
      
      for a in actions:
        q=sum(prob*(reward+gamma*V[next_state]) for prob,next_state,reward in transition_prob[s][a])
        action_value.append(q)

      best_action=np.argmax(action_value)
      policy[s]=best_action
      if old_action != best_action:
        policy_stable=False
    return policy,policy_stable
    
    
def policy_iteration():
  global policy,V 
  while True:
     V=policy_evaluation(policy,V,0.00001)
     policy, stable = policy_improvement(policy, V)
     if stable:
            break
  return policy, V

optimal_policy, optimal_value = policy_iteration()
print("Optimal Policy:", optimal_policy)
print("Optimal Value Function:", optimal_value)

