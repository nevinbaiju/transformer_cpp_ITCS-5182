# Optimizing Attention Layers for efficient Transformer implementation

The repository implements the following parts of implementing transformer layers:

1) A tensor module that handles matrix processing in the CPU as well as GPU.
2) A comprehensive library of various activation and processing functions.
3) Various optimization notes.
4) A comprehensive makefile for building the code for a naive implementation, AVX optimized build, and a CUDA implementation.

## Running the project:

1) Clone the repository.
2) Make sure you have CUDA installed.
3) Run the following commands: ```make normal```, ```make avx```, ```make cuda``` for building the respective implementations.
4) Run the program using ```./run_transformer_{normal/avx/cuda} <num_cols> <num_heads>```.

## Implementation specifics

### Normal

This is a naive implementation using native C++ codes and functions. There is no parallelization used.

### AVX 

AVX intrinsics were used for SIMD processing of data specifically for matrix multiplication. OpenMP was used for parallel processing.
Another optimization tried was switching parallelization between matrix multiplication and individual heads of the attention layer depending on the size.

### CUDA

The codebase was ported to CUDA with kernels for handling different matrix processing functions and activation functions. 
Main optimizations were performed on the matrix multiplication part. Primarily in optimizing the memory access patterns by coalescing and block-tiling the matrix processing. 
Advanced methods of optimizing for operations in a warp can be explored for further optimizations.

## Results

Max Speedup for AVX: ~45
Max Speedup for CUDA: ~160

## Acknowledgement

This project was completed as a part of the course ITCS - 5182 High Performance Computing at UNC Charlotte under the guidance of Dr. Erik Saule.
