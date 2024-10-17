# Nine Men Morris AI
NOTE: This project was tested on Python 3.7, and is discontinued.

A simple Python implementation of the Nine Men Morris game. Uses PythonJIT in order to speed up the performance (for model training) and includes a number of hardcoded and learning AI algorithms that play the game.

Includes the simpler 6-men Morris game for easier testing/evaluation.

Includes a Numpy implementation of a Multilayer Perceptron Neural Net in ```NeuralNet.py```, including initialization, backprop, training, and evaluation. Also includes Genetic algorithm functions such as mutation and copying.

Game implementation is found in the ```nine_men_morris.py``` and ```six_men_morris.py``` python files.

# AI Agents
Also includes a number of AI techniques used to play the game:
## Hardcoded:
```nine_men_solver.py```: Uses a tree search Minimax algorithm to go through all the game scenarios, and then uses a neural network to evaluate the leaf positions. Uses Alpha-Beta pruning and ensures symmetrical and repeating states are not evaluated twice. Uses a very simple and greedy hard-coded evaluation (+100 winning, -100 losing, else equals the difference of pieces between the 2 players)

```minimax_9men.py``` and ```minimax_6men.py```: Also uses a MiniMax algorithm, except a more complicated and intricate evaluation function is used. The function accounts for trapped pieces, as well as "semi-mills" which is a powerful tactic, also gives different values to stones in hand vs. on the playing field.

## Machine Learning:
MinMax with Neural Network Evaluator: Same as MinMax, except the hard-coded evaluation is replaced by a neural net. This network can be trained in 2 different ways: 

```nine_men_net.py```: Uses Reinforcement Learning techniques to train a evaluator neural net. Exploration-exploitation tradeoff is used to both explore novel states and to exploit known good moves in order to produce a well-rounded neural network evaluator.

```minmax_9men_genetic_alg.py```: A genetic algorithm is applied to produce a capable evaluator by intiializing many random nets, then applying evaluation, mutation, and copying, until more capable neural nets emerge.
Training: 



