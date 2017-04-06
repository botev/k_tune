extern crate ocl;
extern crate clap;
use clap::{Arg, App};
use std::str::FromStr;

extern crate k_tune;
use k_tune::gemm;

pub fn gemm(platform_id: usize, device_id: usize,
            m: usize, n: usize, k: usize) -> ::ocl::Result<()> {
    println!("Platform: {}\nDevice: {}\nM: {}\nN: {}\nK: {}\n",
             platform_id,
             device_id,
             m, n, k);
    let params = gemm::GemmBuilder::default()
        .mwg(vec![16, 32, 64, 128])
        .nwg(vec![16, 32, 64, 128])
        .kwg(vec![16, 32])
        .mdimc(vec![8, 16, 32])
        .ndimc(vec![8, 16, 32])
        .mdima(vec![8, 16, 32])
        .ndimb(vec![8, 16, 32])
        .kwi(vec![2, 4, 8])
        .precision(vec![32]).build().unwrap();
    let wrapper = gemm::build_kernel_wrapper("templates/gemm.ocl", m ,n, k);
    let tuner = k_tune::Tuner::new::<&str>(platform_id, device_id, None);
    tuner.tune(wrapper, params, 10);
    Ok(())
}

fn main() {
    let matches = App::new("GEMM Tuner")
        .version("0.1")
        .author("Aleksandar Botev")
        .about("Based on CLTune")
        .arg(Arg::with_name("platform")
            .short("pl")
            .help("Sets the OpenCL platform to use.")
            .default_value("0"))
        .arg(Arg::with_name("device")
            .short("d")
            .help("Sets the OpenCL device to use.")
            .default_value("0"))
        .arg(Arg::with_name("m")
            .help("The first dimension of the matrix A.")
            .default_value("2048"))
        .arg(Arg::with_name("n")
            .help("The second dimension of the matrix B.")
            .default_value("2048"))
        .arg(Arg::with_name("k")
            .help("The second dimension of the matrix A and first dimension of B.")
            .default_value("2048"))
        .get_matches();
    let pid = usize::from_str(matches
        .value_of("platform").unwrap_or("0"))
        .expect("Platform must be a valid integer.");
    let did = usize::from_str(matches
        .value_of("device").unwrap_or("0"))
        .expect("Device must be a valid integer.");
    let m = usize::from_str(matches
        .value_of("m").unwrap_or("2048"))
        .expect("m must be a valid integer.");
    let n = usize::from_str(matches
        .value_of("n").unwrap_or("2048"))
        .expect("n must be a valid integer.");
    let k = usize::from_str(matches
        .value_of("k").unwrap_or("2048"))
        .expect("k must be a valid integer.");
    gemm(pid, did, m, n, k).unwrap();
}
