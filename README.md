# NeuroEvolution of Augmented Topologies Algorithm
This project explores the use of Graph Neural Networks in implementing a Tensorized approach for a variant of genetic algorithms known as NEAT (NeuroEvolution of Augmented Topologies). A double pendulum inversion problem is used as as a benchmark task because of its non-linear nature and because its very cool. 

What differentiates NEAT from other typical genetic algorithms is its ability to optimize model architecture as well as parameters. Built into the mutation function are events that alter the model architecture including:

- Adding a connection between two nodes without forming a loop.
- Removing a current connection.
- Splitting a connection with a newly node.
- Removing a node and all its connections.


As one might imagine, the complexity of maintaining a valid neural network while applying the previous transformations turn this into a challenging architectural and system design problem. Using a tensorized approach and various graph algorithms, the code manages to perform any given mutation on the network in O(1), besides the addConnection() mutation which runs in O(n) where n is the number of nodes due to loop checking. 

This repository includes the code for creating, training, and mutating a Graph Neural Network, calculating and displaying a simualted double pendulum on a moving horizontal pivot, and code to train and store networks on the benchmark problem.

## Inspiration for this project:  

[![YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=9gQQAO4I1Ck)

[![Watch the video](https://img.youtube.com/vi/9gQQAO4I1Ck/hqdefault.jpg)](https://www.youtube.com/watch?v=9gQQAO4I1Ck)


## This project is based on the following paper:  
[![arXiv](https://img.shields.io/badge/arXiv-2404.01817-b31b1b.svg)](https://arxiv.org/abs/2404.01817)


