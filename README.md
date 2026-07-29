# qlearningproject1
# Q-Learning Toy Maintenance Simulator

A simple reinforcement learning project that applies **Q-learning** to a toy maintenance problem.

The environment models a toy that gradually wears out over time. At each step the agent decides whether to:

- Repair the toy
- Replace the toy
- Do nothing

Each action has an associated cost, while using a functioning toy provides a reward. The objective is to maximize the total long-term reward.

---

## How It Works

The environment contains four possible states:

| State | Description |
|-------|-------------|
| 0 | New |
| 1 | Worn |
| 2 | Very Worn |
| 3 | Broken |

The agent learns using the Q-learning update equation

\[
Q(s,a)=Q(s,a)+\alpha[r+\gamma\max_aQ(s',a)-Q(s,a)]
\]

using an epsilon-greedy exploration strategy.

---

## Parameters

- Learning rate (α): 0.1
- Discount factor (γ): 0.8
- Exploration rate (ε): 0.2
- Training episodes: 500

---

## Results

After training, the program outputs:

- Learned Q-table
- Best action for every state
- Performance comparison between the learned policy and a random policy

---

## Technologies

- Python
- NumPy

---

## Future Improvements

- Adjustable degradation probabilities
- Custom reward structures
- Visualization of learning over time
- Hidden Markov Model version of the environment
