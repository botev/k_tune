//use std::ops::Fn;
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::time::{Duration, SystemTime};
use std::ops::*;

use tera;

use ocl::{Platform, Context, Device, Queue,
          Program, Kernel, Buffer, SpatialDims};

#[derive(Clone)]
pub struct ParameterSet {
    pub parameters: HashMap<String, Vec<usize>>,
    pub constraints: Vec<Constraint>,
}

#[derive(Clone, Debug)]
pub struct KernelWrapper {
    pub scalar_inputs: Vec<usize>,
    pub inputs_dims: Vec<(usize, usize)>,
    pub src: String,
    pub name: String,
    pub ref_name: Option<String>
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

        let keys: Vec<String> = params.parameters.keys().cloned().collect();
        let n = keys.len();
        let mut indexes = vec![0; n];
        loop {
            // Fill in parameters
            let mut context = tera::Context::new();
            let config: HashMap<String, usize> = keys
                .iter().cloned()
                .zip(indexes.iter())
                .map(|(key, &i)| {
                    let v = params.parameters[&key][i];
                    context.add(&key, &v);
                    print!("{}={}, ", key, v);
                    (key, v)
                }).collect();
            // Verify constraints
            let mut all_constraints_true = true;
            for constraint in &params.constraints {
                let args: Vec<_> = constraint.args.iter().map(|x| config[x]).collect();
                if ! (constraint.func)(&args) {
                    all_constraints_true = false;
                    break;
                }
            }
            if all_constraints_true {
                // Generate kernel source
                let src = tera::Tera::one_off(&wrapper.src, context, true).unwrap();
                // Run the kernel
                let time = self.run_single_kernel(runs, &src, &buffers, &wrapper);
                // Print time
                println!("Time: {}s.{}ns", time.as_secs(), time.subsec_nanos());
            } else {
                println!("Some constraints are violated for this configuration.")
            }
            // Facilitate iteration over all possible combinations
            let mut last: i32 = indexes.len() as i32 - 1;
            while last >= 0 && indexes[last as usize] ==
                params.parameters[&keys[last as usize]].len() - 1 {
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

    fn run_single_kernel(&self, runs: u32, src: &str,
                         buffers: &[Buffer<f32>],
                         wrapper: &KernelWrapper) -> Duration {
        // Make program
        let program = Program::builder()
            .devices(self.device)
            .src(src)
            .build(&self.context)
            .unwrap();

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
        kernel = kernel.gws(buffers[0].len());

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

type TypeFn = fn(&[usize]) -> bool;

pub struct Constraint {
    pub func: TypeFn,
    pub args: Vec<String>
}

impl Clone for Constraint {
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
