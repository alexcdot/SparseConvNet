# CS179 Project
## Alexander Cui and Yongkyun Lee

### How to run on titan:
`bash develop_titan.sh`

to install required packages, build, and run demos.

The demos print out the pooled output matrices, that match the output we expect from manual calculations.

In particular, the roi pooling demo demonstrates our CPU implementation of max pooling on a matrix of floats with an region-of-interest box that has flexible input size and location, and fixed output dimension.

The matrices and region-of-interest boxes used for the demo are stored in `cpu_demo_data.py`. The code that runs our cpu implementation of roi pooling and compares the results with the expected output is written in `cpu_demo.py`.

### Implementation Details

We implemented ROI pooling in the context of the existing SparseConvNet framwork by adding cpp method along other existing layers (like max pooling, adaptive pooling, and etc.). Receiving the sparse matrices in the format used throughout the library, we calculate the maximum values for each region-of-interest boxes.

### Limitations

The existing approach on sparse tensors uses rulebooks to determine the indices and values of the output matrices. They assume determinisitic output size of (batch, channels, height, width) for max poolilng.

However, for ROI pooling, since the number of region-of-interest boxes are not predetermined and can differ according to different batches of inputs, the ROI pooling could not be implemented using the existing methods. Thus, we came up with 'hacky' methods in terms of indexing and sparse matrix manipulations.

As the level of complexity is very high to develop a fully generalizable ROI pooling in the context of the existing SparseConvNet framework, our approach for GPU implementation is still undecided.

### Repository

The implementation for ROI pooling is uploaded to https://github.com/alexcdot/SparseConvNet. Please check the commit history to check what changes we made on the existing SparseConvNet framework to add ROI pooling.
