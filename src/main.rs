extern crate ocl;
extern crate rand;
extern crate k_tune;

use k_tune::gemm;
use k_tune::simple;

pub fn simple() -> ::ocl::Result<()> {
    let params = simple::SimpleBuilder::new()
        .value(vec![8, 16, 32])
        .build().unwrap();
    let wrapper = simple::build_kernel_wrapper(1024, 1024);
    let mut tuner = k_tune::Tuner::new("templates/*", 0, 0);
    tuner.tune(wrapper, params, 10);
    Ok(())
}

pub fn gemm() -> ::ocl::Result<()> {
    let params = gemm::GemmBuilder::default().build().unwrap();
    let wrapper = gemm::build_kernel_wrapper(1024, 1024, 1024);
    let mut tuner = k_tune::Tuner::new("templates/*", 0, 0);
    tuner.tune(wrapper, params, 10);
    Ok(())
}

fn main() {
    simple().unwrap();
}