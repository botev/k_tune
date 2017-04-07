use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::time::Duration;
use std::ops::Index;
use std::io::Write;

use ocl::{Platform, Context, Device, Queue, Event,
          Program, Kernel, Buffer, SpatialDims};
use ocl::flags::CommandQueueProperties;
use ocl::enums::ProfilingInfo;
use futures::future::Future;


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
    pub scalar_inputs: Vec<i32>,
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
    queue: Queue
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
        println!("Platform: {} - {}", platform.name(), platform.version());
        println!("Device: {} by {}", device.name(), device.vendor());
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .unwrap();
        let queue_flags = Some(CommandQueueProperties::new().profiling());
        let queue = Queue::new(&context, device, queue_flags).unwrap();
        Tuner {
            device: device,
            context: context,
            queue: queue,
        }
    }

    pub fn tune(&self, wrapper: KernelWrapper,
                params: ParameterSet,
                runs: u32, log_file: Option<&str>) {
        let mut log_file = log_file.map(|x| ::std::fs::File::create(x).unwrap());
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

        if let Some(f) = log_file.as_mut() {
            Tuner::write_header(&params.parameters, f).expect("Writing header failed.");
        } else {
            Tuner::print_header(&params.parameters);
        }

        let mut indexes = vec![0; params.len()];
        loop {
            // Fill in parameters
            //            let mut context = tera::Context::new();
            let config: HashMap<String, i32> = params
                .parameters.iter()
                .zip(indexes.iter())
                .map(|(&(ref key, ref values), &i)| {
                    let v = values[i];
                    //                    context.add(&key, &v);
//                    print!("|{:>10}", v);
                    (key.clone(), v)
                }).collect();

            // Verify constraints
            let mut failed = None;
            for (i, constraint) in params.constraints.iter().enumerate() {
                let args: Vec<_> = constraint.args.iter().map(|&x| config[x]).collect();
                if ! (constraint.func)(&args) {
                    failed = Some(i);
                    break;
                }
            }
            if failed.is_none() {
                // Run the kernel
                let time = self.run_single_kernel(runs, &wrapper, &params, &config, &buffers);
                // Configuration parameters in order
                let ordered = params.parameters.iter().map(|&(ref k, _)| config[k]).collect();
                // Print time
                if let Some(f) = log_file.as_mut() {
                    Tuner::write_parameters(ordered, time, f).expect("Writing parameters failed.");
                } else {
                    Tuner::print_parameters(ordered, time);
                }
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
                         params: &ParameterSet,
                         config: &HashMap<String, i32>,
                         buffers: &[Buffer<f32>]) -> Duration {
        let (gws, lws) = Tuner::calculate_work_sizes(&wrapper, &params, &config);

        // Build the program with all defines
        let mut program = Program::builder();
        for (& ref k, & ref v) in config.iter() {
            program = program.cmplr_def(k.clone(), *v);
        }
        let program = program
            .devices(self.device)
            .src(wrapper.src.clone())
            .build(&self.context).unwrap();

        // Make kernel
        let mut kernel = Kernel::new(wrapper.name.clone(), &program)
            .unwrap()
            .queue(self.queue.clone())
            .gws(gws)
            .lws(lws);

        // Add arguments
        for &i in &wrapper.scalar_inputs {
            kernel = kernel.arg_scl(i);
        }
        for & ref buffer in buffers {
            kernel = kernel.arg_buf(buffer);
        }

        // Run the kernel
        let mut times = Vec::new();
        for _ in 0..runs {
            // Event for timing
            let mut kernel_event = Event::empty();
            kernel.cmd()
                .enew(&mut kernel_event)
                .enq().unwrap();
            kernel_event.clone().wait().unwrap();
            let command_start: u64 = kernel_event.profiling_info(ProfilingInfo::Start)
                .time().unwrap();
            let command_end: u64 = kernel_event.profiling_info(ProfilingInfo::End)
                .time().unwrap();
            let time = command_end - command_start;
            times.push(Duration::new(time / 1000000000, (time % 1000000000) as u32));
        }
        times.iter().sum::<Duration>() / runs
    }

    fn print_header(parameters: &Vec<(String, Vec<i32>)>) {
        for &(ref k, _) in parameters {
            if k.len() > 8 {
                print!("|{:^8}", &k[0..8]);
            } else {
                print!("|{:^8}", k);
            }
        }
        println!("|{:^13}|", "Time(s.ns)");
        let l = 9 * parameters.len() + 15;
        println!("{}", (0..l).map(|_| "-").collect::<String>());
    }

    fn write_header(parameters: &Vec<(String, Vec<i32>)>, f: &mut Write)
        -> ::std::io::Result<()> {
        for &(ref k, _) in parameters {
            if k.len() > 8 {
                write!(f, "{:^8}, ", &k[0..8])?;
            } else {
                write!(f, "{:^8}, ", k)?;
            }
        }
        writeln!(f, "{:^13}", "Time(s.ns)")
    }

    fn print_parameters(parameters: Vec<i32>, time: Duration) {
        // Print time
        for value in parameters {
            print!("|{:>8}", value);
        }
        println!("|{:>3}.{:<09}|", time.as_secs(), time.subsec_nanos());
    }

    fn write_parameters(parameters: Vec<i32>, time: Duration, f: &mut Write)
        -> ::std::io::Result<()> {
        // Print time
        for value in parameters {
            write!(f, "{:>8}, ", value)?;
        }
        writeln!(f, "{:>3}.{:<09}", time.as_secs(), time.subsec_nanos())
    }

    fn calculate_work_sizes(wrapper: &KernelWrapper,
                            params: &ParameterSet,
                            config: &HashMap<String, i32>) -> (SpatialDims, SpatialDims) {
        let mut global_size = wrapper.global_base;
        let mut local_size = wrapper.local_base;
        if global_size.dim_count() != local_size.dim_count() {
            panic!("Different number of dimensions of global_size and local_size.")
        }
        if let Some(ref mul) = params.mul_global_size {
            if mul.len() != global_size.dim_count() as usize {
                panic!("Different number of multipliers for global_size")
            }
            let mul: Vec<i32> = mul.iter().map(|x| {
                match *x {
                    Some(ref key) => config[key],
                    None => 1
                }
            }).collect();
            global_size = match global_size {
                SpatialDims::Unspecified => panic!("Unspecified spatial dims"),
                SpatialDims::One(x) => SpatialDims::One(x * mul[0] as usize),
                SpatialDims::Two(x, y) => SpatialDims::Two(x * mul[0] as usize,
                                                           y * mul[1] as usize),
                SpatialDims::Three(x, y, z) =>
                    SpatialDims::Three(x * mul[0] as usize,
                                       y * mul[1] as usize,
                                       z * mul[2] as usize)
            }
        }
        if let Some(ref mul) = params.mul_local_size {
            if mul.len() != local_size.dim_count() as usize {
                panic!("Different number of multipliers for local_size")
            }
            let mul: Vec<i32> = mul.iter().map(|x| {
                match *x {
                    Some(ref key) => config[key],
                    None => 1
                }
            }).collect();
            local_size = match local_size {
                SpatialDims::Unspecified => panic!("Unspecified spatial dims"),
                SpatialDims::One(x) => SpatialDims::One(x * mul[0] as usize),
                SpatialDims::Two(x, y) => SpatialDims::Two(x * mul[0] as usize,
                                                           y * mul[1] as usize),
                SpatialDims::Three(x, y, z) =>
                    SpatialDims::Three(x * mul[0] as usize,
                                       y * mul[1] as usize,
                                       z * mul[2] as usize)
            }
        }
        if let Some(ref div) = params.div_global_size {
            if div.len() != global_size.dim_count() as usize {
                panic!("Different number of multipliers for local_size")
            }
            let div: Vec<i32> = div.iter().map(|x| {
                match *x {
                    Some(ref key) => config[key],
                    None => 1
                }
            }).collect();
            global_size = match global_size {
                SpatialDims::Unspecified => panic!("Unspecified spatial dims"),
                SpatialDims::One(x) => SpatialDims::One(x / div[0] as usize),
                SpatialDims::Two(x, y) => SpatialDims::Two(x / div[0] as usize,
                                                           y / div[1] as usize),
                SpatialDims::Three(x, y, z) =>
                    SpatialDims::Three(x / div[0] as usize,
                                       y / div[1] as usize,
                                       z / div[2] as usize)
            }
        }
        (global_size, local_size)
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
