use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

use core::*;
use ocl::SpatialDims;

pub fn build_kernel_wrapper(file: &str, m: usize, n: usize) -> KernelWrapper {
    let mut src = String::new();
    File::open(file)
        .unwrap()
        .read_to_string(&mut src)
        .unwrap();
    KernelWrapper {
        scalar_inputs: vec![],
        inputs_dims: vec![(m, n), (m, n), (m, n)],
        src: src,
        name: "add".into(),
        ref_name: None,
        global_base: SpatialDims::Two(m, n),
        local_base: SpatialDims::Two(1, 1),
    }
}

#[derive(Clone, Debug)]
pub struct SimpleBuilder {
    parameters: HashMap<String, Vec<i32>>,
}

impl Default for SimpleBuilder {
    fn default() -> Self {
        SimpleBuilder::new().value1(vec![2]).value2(vec![2])
    }
}

impl SimpleBuilder {
    pub fn new() -> Self {
        SimpleBuilder { parameters: HashMap::new() }
    }

    pub fn value1(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("VALUE1".into(), values);
        self
    }

    pub fn value2(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("VALUE2".into(), values);
        self
    }

    pub fn build<'a>(self) -> Result<ParameterSet<'a>, String> {
        let ordered = vec!["VALUE1", "VALUE2"];
        for &name in &ordered {
            if self.parameters.get(name).is_none() {
                return Err(format!("The Simple parameter set for '{}' has not been set.", name));
            }
        }
        let parameters = ordered
            .iter()
            .map(move |&x| {
                     let s: String = x.into();
                     let v = self.parameters[&s].clone();
                     (s, v)
                 })
            .collect();
        let mut constraints: Vec<FnWrap<'static, bool>> = Vec::new();
        fn multiple_of_x(v: &[i32]) -> bool {
            v[1] % v[0] == 0
        };
        constraints.push(FnWrap {
                             func: multiple_of_x,
                             args: vec!["VALUE1", "VALUE2"],
                         });
        Ok(ParameterSet {
               parameters: parameters,
               constraints: constraints,
               local_memory_needed: None,
               mul_global_size: None,
               mul_local_size: None,
               div_global_size: None,
           })
    }
}
