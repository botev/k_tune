#!/bin/bash
cargo run --release --example gemm -- -f gemm_gpu.csv -d 0
cargo run --release --example gemm -- -f gemm_cpu.csv -d 1
