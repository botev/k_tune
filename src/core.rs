//use std::ops::Fn;
use std::collections::HashMap;
use std::boxed::Box;
use rand::{thread_rng, Rng};
use std::time::{Duration, SystemTime};

use tera;

use ocl::{Platform, Context, Device, Queue,
          Program, Kernel, Buffer, SpatialDims};

#[derive(Default, Clone)]
pub struct ParameterSet {
    pub parameters: HashMap<String, Vec<usize>>,
//    pub constraints: Vec<Box<Fn(Vec<usize>)->bool>>,
}

#[derive(Default, Clone)]
pub struct KernelWrapper {
    pub scalar_inputs: Vec<usize>,
    pub inputs_dims: Vec<(usize, usize)>,
    pub name: String,
    pub src: String,
    pub reference_src: Option<String>
}

pub struct Tuner {
    device: Device,
    context: Context,
    queue: Queue,
    rng: Box<Rng>,
    folder: String
}

impl Tuner {
    pub fn new(folder: &str, platform_id: usize, device_id: usize) -> Self {
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
            rng: Box::new(thread_rng()),
            folder: String::new() + folder
        }
    }

    pub fn tune(&mut self, wrapper: KernelWrapper, params: ParameterSet, runs: u32) {
        // Generate buffers
        let buffers: Vec<Buffer<f32>> = wrapper.inputs_dims.iter().map(|&(ref r, ref c)|
            Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .dims(SpatialDims::Two(*r, *c))
                .build().unwrap()
        ).collect();
        // Populate buffers
        for buf in &buffers {
            let vec = self.rng.gen_iter::<f32>().take(buf.len()).collect::<Vec<f32>>();
            buf.write(&vec).enq().unwrap();
        }
        // Todo make this a grid
        let key = "VALUE";
        for value in params.parameters.get(key).unwrap() {
            // Generate kernel source
            let mut context = tera::Context::new();
            context.add(key, value);
            let src = tera::Tera::new(&self.folder).unwrap()
                .render(&wrapper.src, context).unwrap();

            // Make program
            let program = Program::builder()
                .devices(self.device)
                .src(src)
                .build(&self.context)
                .unwrap();

            // Make kernel
            let mut kernel = Kernel::new(wrapper.name.clone(), &program)
                .unwrap()
                .queue(self.queue.clone());

            // Add arguments
            for &i in &wrapper.scalar_inputs {
                kernel = kernel.arg_scl(i);
            }
            for & ref buffer in &buffers {
                kernel = kernel.arg_buf(buffer);
            }
            // Todo
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
            // Print first 10 elements of all buffers
            println!("VALUE: {}, Average time: {}s.{}ns",
                     value, average.as_secs(), average.subsec_nanos());
            for & ref buffer in &buffers {
                let mut vec = vec![0.0f32; buffer.len()];
                buffer.read(&mut vec).enq().unwrap();
                println!("{:?}", &vec[0..10]);
            }
        }

    }

}
