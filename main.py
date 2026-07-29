import numpy as np
import random

#game where states defined as
# 0 = new, 1 = worn, 2 = very worn, 3 = broken
#actions defined as
# 0 = repair 1 = replace
#rewards defined as 
# repair cost = -2 replace cost = -6
# working toy reward = +5 broken penalty = -10

#find repair increment
#repair_increment = input("Enter repair increment")

#find p of degredation 
#p = input("Enter a number between (0,1) for the percentage chance of toy degredation each step.")

#states
states = [0, 1, 2, 3]

#actions
REPAIR = 0
REPLACE = 1
NEITHER = 2
actions = [REPAIR, REPLACE, NEITHER]

#Q-tables (4 states x 3 actions)
Q = np.zeros((4,3))

#Parameters
alpha = 0.1 #learning rate
gamma = 0.8 #discount factor
epsilon = 0.2 #exploration

episodes = 500 

#state function takes action -> reward 
#returns the next state and the reward received
def step(state, action):
    reward = 0

    #if modified toy
    if action == REPAIR:
        reward -= 2
        next_state = max(0, state -1)

    elif action == REPLACE:
        reward -= 8
        next_state = 0 #brand new

    elif action == NEITHER:
        next_state = state #no change

    #if user used toy
    if next_state < 3:
        reward += 5
    else:
        reward -= 15

    #Random chance the toy becomes more worn after each use
    if random.random() <= 0.3:
        next_state = min(next_state +1 , 3)
    
    return next_state, reward

for episode in range(episodes):

    state = random.choice(states)

    for step_count in range(20): #20 steps per episodes

        #epsilon-greedy action
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward = step(state, action)

        # Update the Q-value using the Bellman equation
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

print("learned Q-tables:")
print(Q)

print("\nBest policy:")
for s in states:
    action = np.argmax(Q[s])
    action_name = ["Repair", "Replace", "Neither"][action]
    print(f"State {s}: {action_name}")


# Compare the learned policy against a completely random policy
eval_steps = 1000

#Learned policy
learned_reward = 0 
state = random.choice(states)
for _ in range(eval_steps):
    action = np.argmax(Q[state])
    state,reward = step(state, action)
    learned_reward += reward

#Random policy
random_reward = 0
state = random.choice(states)
for _ in range(eval_steps):
    action = random.choice(actions)
    state,reward = step(state, action)
    random_reward += reward

print("\nPolicy Comparison (over {} steps):".format(eval_steps))
print(f"Learned policy total reward: {learned_reward}")
print(f"Random policy total reward: {random_reward}")
print(f"Improvement: {learned_reward - random_reward} ({((learned_reward - random_reward) / abs(random_reward)) * 100:.1f}%)")
