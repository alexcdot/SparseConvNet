# CS179 Project
## Alexander Cui and Yongkyun Lee

### How to run on titan:
`bash develop_titan.sh`

to install required packages, build, and run demos.

The demos print out the pooled output matrices, that match the output we expect from manual calculations.

In particular, the roi pooling demo demonstrates our CPU implementation of max pooling on a matrix of floats with an region-of-interest box that has flexible input size and location, and fixed output dimension.