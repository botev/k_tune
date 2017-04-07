# k_tune

A kernel tuning library for OpenCL. 
Inspired and based on [CLTune](https://github.com/CNugteren/CLTune), 
rewritten in pure Rust using the [ocl](https://github.com/cogciprocate/ocl).
 
## Goals

The library provide an easy way for running and timing an OpenCL kernels with
extra parameters (these are not inputs, but hyper parameters). Standard examples
of such parameters are threads per work group as well as local and global work
group sizes. The library also provides a way of specifying constraints on the 
parameters, such that any invalid configurations on the grid to be skipped.


## Examples

Currently the repository has only a single example for auto tuning a GEMM kernel.

## License

Apache License, Version 2.0