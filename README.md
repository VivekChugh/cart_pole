# cart_pole
Simple Reinforcement learning agent that learns balance a pole on cart by moving left or right.

I have put together a project over weekend. 
- It implements a reinforcement learning agent.
- learning agent learns to take action (moving cart left or right) based to it prior experiences.
- It samples batch of previous experiences from a replay memory.
- It uses primitive two nural-nets, one as policy network and other target network.
- Follows Epsilon greedy strategy to choose between exploration and exploitation at each time step
- Updates weights and biases of target net from policy net every 10th episode
  
NOTE:  while sampling experience batch from replay memory and creating state tensors/action tensors, I may have messed up their dimensions before of passing to policy net which causing program to break after 256 time steps in total (when main program accesses the policy net for first time with sampled batch of state-action pairs from replay memory). I'll will be working on it.
    
