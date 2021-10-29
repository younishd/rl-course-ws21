# Lab 5

In this lab we look into implementations of Deep Q-Networks (DQNs) in PyTorch.

### Task 1:
Check the file [`5_DQN_LunarLander.ipynb`](5_DQN_LunarLander.ipynb)
which implements a DQN on the [LunarLander](https://gym.openai.com/envs/LunarLander-v2/) environment.

1. Execute the code on your machine (or on Google Colab) and watch the recorded video.
2. Read through the code and then start to understand how the DQN is implemented.
    - Hint 1: Start with section 3. in the notebook, which starts the training process.
    - Hint 2: Put plenty of `print` statements in the code to see what is insides the variables.
3. Answer the following questions about the code:
    - What dimension is the input and output tensor of the neural network and what does every dimension mean?
        1. In the `act`-method of the client?
        2. In the `learn`-method of the client?
    - How does a state look in this environment?
    - In which cases does the loop in cell 9 terminate?
    - How often is a training step of the network performed?
    - On how many data points is the network trained in one training step?
    - Which role does the `done`-flag play during learning?
    - Which values are contained in the tensor `Q_excpected` in the `learn`-method?
    - What is the role of `eps` and how does it evolve? Plot its value over time.

### Task 2:
Create a new notebook in which you solve the [CartPole](https://gym.openai.com/envs/CartPole-v0/)
environment using a DQN.
- Use the previous notebook as starting point and train until a mean reward of 200 is achieved.
- Play one episode with the trained model and record the outcome.

### Bonus Task:
Go through [this](5_DQN_Pong.ipynb) implementation of a DQN for
playing Atari Pong directly on pixels values.
Run the notebook on your machine (or Colab) and try to understand what is going on.



