extern crate ocl;

extern crate k_tune;
use k_tune::simple;

pub fn simple() -> ::ocl::Result<()> {
    let params = simple::SimpleBuilder::new()
        .value1(vec![8, 16])
        .value2(vec![8, 16, 32])
        .build()
        .unwrap();
    let wrapper = simple::build_kernel_wrapper("templates/simple.ocl", 1024, 1024);
    let tuner = k_tune::Tuner::default();
    tuner.tune(wrapper, params, 10, None);
    Ok(())
}

fn main() {
    simple().unwrap();
}
