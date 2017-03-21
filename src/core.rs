//use std::ops::Fn;
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::time::{Duration, SystemTime};
use std::ops::{Index};

//use tera;

use ocl::{Platform, Context, Device, Queue,
          Program, Kernel, Buffer, SpatialDims};

#[derive(Clone)]
pub struct ParameterSet<'a> {
    pub parameters: Vec<(String, Vec<i32>)>,
    pub constraints: Vec<Constraint<'a>>,
    pub mul_local_size: Option<Vec<Option<String>>>,
    pub mul_global_size: Option<Vec<Option<String>>>,
    pub div_global_size: Option<Vec<Option<String>>>,
}

impl<'a, 'b> Index<&'b str> for ParameterSet<'a> {
    type Output = Vec<i32>;
    fn index(&self, index: &'b str) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<'a> ParameterSet<'a> {
    fn get(&self, key: &str) -> Result<&Vec<i32>, String> {
        for &(ref k, ref v) in &self.parameters {
            if k == key {
                return Ok(v)
            }
        }
        Err(format!("Key {} does not exist.", key))
    }

    fn len(&self) -> usize {
        self.parameters.len()
    }
}

#[derive(Clone, Debug)]
pub struct KernelWrapper {
    pub scalar_inputs: Vec<usize>,
    pub inputs_dims: Vec<(usize, usize)>,
    pub src: String,
    pub name: String,
    pub ref_name: Option<String>,
    pub global_base: SpatialDims,
    pub local_base: SpatialDims
}

#[derive(Clone, Debug)]
pub struct Tuner {
    device: Device,
    context: Context,
    queue: Queue,
}

impl Default for Tuner{
    fn default() -> Self {
        Tuner::new(0, 0)
    }
}

impl Tuner {
    pub fn new(platform_id: usize, device_id: usize) -> Self {
        let platform = Platform::list()[platform_id];
        let device = Device::list_all(&platform).unwrap()[device_id];
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .unwrap();
        let queue = Queue::new(&context, device, None).unwrap();
        Tuner {
            device: device,
            context: context,
            queue: queue,
        }
    }

    pub fn tune(&self, wrapper: KernelWrapper, params: ParameterSet, runs: u32) {
        // Generate buffers
        let buffers: Vec<Buffer<f32>> = wrapper.inputs_dims.iter().map(|&(ref r, ref c)|
            Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .dims(SpatialDims::Two(*r, *c))
                .build().unwrap()
        ).collect();
        // Populate buffers
        let mut rng = thread_rng();
        for buf in &buffers {
            let vec = rng.gen_iter::<f32>().take(buf.len()).collect::<Vec<f32>>();
            buf.write(&vec).enq().unwrap();
        }

        let mut indexes = vec![0; params.len()];
        for &(ref k, _) in &params.parameters {
            print!("|{:^12}", k);
        }
        println!("|{:^17}|", "Time(s.ns)");
        let l = 13 * params.parameters.len() + 19;
        println!("{}", (0..l).map(|_| "-").collect::<String>()) ;
        loop {
            // Fill in parameters
            //            let mut context = tera::Context::new();
            let config: HashMap<String, i32> = params
                .parameters.iter()
                .zip(indexes.iter())
                .map(|(&(ref key, ref values), &i)| {
                    let v = values[i];
                    //                    context.add(&key, &v);
                    print!("|{:>12}", v);
                    (key.clone(), v)
                }).collect();

            // Verify constraints
            let mut all_constraints_true = true;
            for constraint in &params.constraints {
                let args: Vec<_> = constraint.args.iter().map(|&x| config[x]).collect();
                if ! (constraint.func)(&args) {
                    all_constraints_true = false;
                    break;
                }
            }
            if all_constraints_true {
                //                // Generate kernel source
                //                let src = tera::Tera::one_off(&wrapper.src, context, true).unwrap();
                // Run the kernel
                let time = self.run_single_kernel(runs, &wrapper, config, &buffers);
                // Print time
                println!("|{:>8}.{:<8}|", time.as_secs(), time.subsec_nanos());
            } else {
                println!("|{:>8}.{:<8}|", "-", "----");
            }
            // Facilitate iteration over all possible combinations
            let mut last: i32 = indexes.len() as i32 - 1;
            while last >= 0 && indexes[last as usize] ==
                params.parameters[last as usize].1.len() - 1 {
                last -= 1;
            }
            if last == -1 {
                break;
            }
            indexes[last as usize] += 1;
            for i in indexes.iter_mut().skip(last as usize + 1) {
                *i = 0;
            }
        }
    }

    fn run_single_kernel(&self, runs: u32,  wrapper: &KernelWrapper,
                         config: HashMap<String, i32>,
                         buffers: &[Buffer<f32>]) -> Duration {
        // Calculate global and local size
        let mut global_size = wrapper.global_base;
        let mut local_size = wrapper.local_base;

        // Build the program with all defines
        let mut program = Program::builder();
        println!("****");
        for (k, v) in config.into_iter() {
            println!("#define {} {}", k, v);
            program = program.cmplr_def(k, v);
        }
        let program = program
            .devices(self.device)
            .src_file(wrapper.src.clone())
            .build(&self.context).unwrap();

        // Make kernel
        let mut kernel = Kernel::new(wrapper.name.clone(), &program)
            .unwrap().queue(self.queue.clone());

        // Add arguments
        for &i in &wrapper.scalar_inputs {
            kernel = kernel.arg_scl(i);
        }
        for & ref buffer in buffers {
            kernel = kernel.arg_buf(buffer);
        }

        // Todo calculate this better
        kernel = kernel.gws(global_size).lws(local_size);

        // Run the kernel
        let mut times = Vec::new();
        for _ in 0..runs {
            let start = SystemTime::now();
            kernel.enq().unwrap();
            times.push(start.elapsed().unwrap());
        }
        let average = times.into_iter()
            .fold(Duration::new(0, 0), |sum, val| sum + val) / runs;
        average
    }
}

type TypeFn = fn(&[i32]) -> bool;

pub struct Constraint<'a> {
    pub func: TypeFn,
    pub args: Vec<&'a str>
}

impl<'a> Clone for Constraint<'a> {
    fn clone(&self) -> Self {
        Constraint {
            func: self.func,
            args: self.args.clone()
        }
    }
}

pub fn is_power_of_two(value: &usize) -> bool {
    value & (value - 1) == 0
}
