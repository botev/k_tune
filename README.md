# k_tune

A kernel tuning library for OpenCL. 

Based on [CLTune](https://github.com/CNugteren/CLTune), 
rewritten in pure Rust using the [ocl](https://github.com/cogciprocate/ocl) crate.
 
 
## Goals

The library provide an easy way for running and timing OpenCL kernels with
hyper parameters (parameters which are not inputs the self). 
The library also provides a way of specifying constraints on the 
parameters, such that any invalid configurations on the grid can be skipped,
as well as functions hooks for calculating the required local memory for 
skipping configurations which go over the maximum allowed on the device.

## Why rewrite CLTune?

There are two main reasons I wanted to do this. 

First, I was writing a library in Rust which uses OpenCL and needs to be 
as fast as possible. 

Second, I do like a lot more the safe wrappers of the [ocl](https://github.com/cogciprocate/ocl)
crate around OpenCL handles and manipulation. Additionally, I think Cargo and Rust
provide a much more pleasant experience around downloading and running code 
compare to C++.

Finally, I like programming in Rust a lot and this looked like a great way
to implement something useful. 


## Examples

Currently the repository has only a single example.
 
### General Matrix Multiplication (GEMM)

The kernel has the following parameters:

* `MWG` - Tile-size in dimension M (e.g. 64, 128)
* `NWG` - Tile-size in dimension N (e.g. 64, 128)
* `KWG` - Tile-size in dimension K (e.g. 8, 16)
* `MDIMC` - Threads per workgroup in M-dimension (e.g. 8, 16, 32)
* `NDIMC` - Threads per workgroup in N-dimension (e.g. 8, 16, 32)
* `MDIMA` - Re-shaped tile dimension of matrix A: KDIMA * MDIMA
* `NDIMB` - Re-shaped tile dimension of matrix B: KDIMB * NDIMB
* `KWI` - Unroll factor of the KWG loop (smaller or equal than KWG)
* `VWM` - Vector width of matrices A and C (supported 1, 2, 4, and 8)
* `VWN` - Vector width of matrix B (supported 1, 2, 4, and 8)
* `STRM` - Use strided access within a thread in the M-dimension (1) or not (0)
* `STRN` - Use strided access within a thread in the N-dimension (1) or not (0)
* `SA` - Use local/shared memory to cache matrix A (1) or not (0)
* `SB` - Use local/shared memory to cache matrix B (1) or not (0)
* `PRECISION` - Whether to use single (32) or double (64) precision data-types

There are seven constraints that the parameters must satisfy to be valid:

* `KWG % KWI == 0`
* `MWG % (MDIMC * VWM) == 0`
* `NWG % (NDIMC * VWN) == 0`
* `MWG % (MDIMA * VWM) == 0`
* `NWG % (NDIMB * VWN) == 0`
* `KWG % ((MDIMC * NDIMC) / MDIMA) == 0`
* `KWG % ((MDIMC * NDIMC) / NDIMB) == 0`

The local memory required by a parameter configuration is: 

`SA * KWG * (MWG / VWM) + SB * KWG * (NWG / VWN)`

## License

Apache License, Version 2.0