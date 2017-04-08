extern crate ocl;
extern crate clap;
use clap::{Arg, App};
use std::str::FromStr;

extern crate k_tune;
use k_tune::gemm;

pub fn gemm(platform_id: usize, device_id: usize,
            m: usize, n: usize, k: usize,
            file: Option<&str>, runs: usize) -> ::ocl::Result<()> {
    println!("Platform: {}\nDevice: {}\nM: {}\nN: {}\nK: {}\n",
             platform_id,
             device_id,
             m, n, k);
    let params = gemm::GemmBuilder::default()
        .mwg(vec![256, 128, 64, 32])
        .nwg(vec![256, 128, 64, 32])
        .kwg(vec![32, 16, 8])
        .mdimc(vec![32, 16, 8])
        .ndimc(vec![32, 16, 8])
        .mdima(vec![32, 16, 8])
        .ndimb(vec![32, 16, 8])
        .kwi(vec![16, 8, 4, 2])
        .precision(vec![32]).build().unwrap();
    let wrapper = gemm::build_kernel_wrapper("templates/gemm.ocl", m ,n, k);
    let tuner = k_tune::Tuner::new(platform_id, device_id);
    tuner.tune(wrapper, params, runs, file);
    Ok(())
}

fn main() {
    let matches = App::new("GEMM Tuner")
        .version("0.1")
        .author("Aleksandar Botev")
        .about("Based on CLTune")
        .arg(Arg::with_name("file")
            .short("f").long("file")
            .takes_value(true)
            .help("The log file to which to write results."))
        .arg(Arg::with_name("runs")
            .short("r").long("runs")
            .takes_value(true)
            .default_value("10")
            .help("The log file to which to write results."))
        .arg(Arg::with_name("platform")
            .short("p").long("platform")
            .takes_value(true)
            .default_value("0")
            .help("Sets the OpenCL platform to use."))
        .arg(Arg::with_name("device")
            .short("d").long("device")
            .takes_value(true)
            .default_value("0")
            .help("Sets the OpenCL device to use."))
        .arg(Arg::with_name("m")
            .short("m").long("m")
            .takes_value(true)
            .default_value("2048")
            .help("The first dimension of the matrix A."))
        .arg(Arg::with_name("n")
            .short("n").long("n")
            .takes_value(true)
            .default_value("2048")
            .help("The second dimension of the matrix B."))
        .arg(Arg::with_name("k")
            .short("k").long("k")
            .takes_value(true)
            .default_value("2048")
            .help("The second dimension of the matrix A and first dimension of B."))
        .get_matches();
    let pid = usize::from_str(matches
        .value_of("platform").unwrap())
        .expect("Platform must be a valid integer.");
    let did = usize::from_str(matches
        .value_of("device").unwrap())
        .expect("Device must be a valid integer.");
    let m = usize::from_str(matches
        .value_of("m").unwrap())
        .expect("m must be a valid integer.");
    let n = usize::from_str(matches
        .value_of("n").unwrap())
        .expect("n must be a valid integer.");
    let k = usize::from_str(matches
        .value_of("k").unwrap())
        .expect("k must be a valid integer.");
    let file = matches.value_of("file");
    let runs = usize::from_str(matches.value_of("runs").unwrap()).unwrap();
    gemm(pid, did, m, n, k, file, runs).unwrap();
}
