# Tetris AI using Reinforcement Learning

The main objective of this project is to train an Agent to achieve the best possible score in the Tetris game 

The Agent is trained using a Deep Q-network (DQN), a Q-Learning variant based on a neural network to approximate Q function, when the number of possible states becomes too big to fit in a Q table.

<img src="./assets/tetris.gif" alt="screencast" width="400"/>


## description

the input of the NN is a simplified game state, instead of the 1-bit grid. This is to drastically reduce the training time and the neural network architecture (a grid would need a more complex Convoluted network)
The state consists in the list of heights and holes.
The output returns the Q-Value for a state

So this approach is an "afterstate" evaluation of Q : we have to test the network with all possible states from a current state (max 40), and choose the best Q.

This differs from the Atari classic DQN where the network ouputs directly the action to play.
I've tried this with different network architecture, but it didn't played well..
I think it would require much more training

NB : for stability of training, there is a second DQN updated every 500 steps (The loss of next state is calculated from the second DQN and back propagated to the first network)

## usage
### install
```
pip install -r requirements.txt
```

### run the Snake agent with the current model 
```
python3 ui.py
```

### train a new agent
```
python3 train.py
```
