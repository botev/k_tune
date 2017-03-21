extern crate ocl;
extern crate rand;
extern crate k_tune;

use k_tune::gemm;
use k_tune::simple;

pub fn simple() -> ::ocl::Result<()> {
    let params = simple::SimpleBuilder::new()
        .value1(vec![8, 16])
        .value2(vec![8, 16, 32])
        .build().unwrap();
    let wrapper = simple::build_kernel_wrapper("templates/simple.ocl", 1024, 1024);
    let tuner = k_tune::Tuner::default();
    tuner.tune(wrapper, params, 10);
    Ok(())
}

pub fn gemm() -> ::ocl::Result<()> {
    let params = gemm::GemmBuilder::default().build().unwrap();
    let wrapper = gemm::build_kernel_wrapper("templates/gemm.ocl", 1024, 1024, 1024);
    let tuner = k_tune::Tuner::default();
    tuner.tune(wrapper, params, 10);
    Ok(())
}

fn main() {
    simple().unwrap();
}