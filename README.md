# Meta-Learning-LR
Using a meta learning approach to predict the optimal learning rate for a network


# Getting Started

## Setup
For a quick setup follow the next steps:

conda create -n meta python=3.10.4

conda activate meta

git clone https://github.com/Shaier/Meta-Learning-LR.git

cd Meta-Learning-LR

pip install -r requirements.txt


## Running the code
There are 2 main files:
1. [simple neural network file](simple_neural_network.py) which can be run using "python simple_neural_network.py" from the terminal.
This is just a simple FC neural network. To get different saved predictions change the learning rate on line 60.
2. [meta training file](lr_per_n_training_meta_function.ipynb), which can be run by going into the file and running it (it's a jupyter notebook).
This is the file that trains a meta learning model to output an optimal learning rate for the simple NN.
