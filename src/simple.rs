use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

use core::*;

pub fn build_kernel_wrapper(file: &str, m: usize, n: usize) -> KernelWrapper {
    let mut src = String::new();
    File::open(file).unwrap().read_to_string(&mut src).unwrap();
    KernelWrapper {
        scalar_inputs: vec![],
        inputs_dims: vec![(m, n), (m, n), (m, n)],
        src: src,
        name: "add".into(),
        ref_name: None
    }
}

#[derive(Clone, Debug)]
pub struct SimpleBuilder {
    parameters: HashMap<String, Vec<usize>>,
}

impl Default for SimpleBuilder {
    fn default() -> Self {
        SimpleBuilder::new()
            .value1(vec![2])
            .value2(vec![2])
    }
}

impl SimpleBuilder {
    pub fn new() -> Self {
        SimpleBuilder{parameters: HashMap::new()}
    }

    pub fn value1(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("VALUE1".into(), values);
        return self
    }

    pub fn value2(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("VALUE2".into(), values);
        return self
    }

    pub fn build(self) -> Result<ParameterSet, String> {
        if self.parameters.get("VALUE1").is_none() {
            return Err("The Simple parameter set for 'VALUE1' has not been set.".into())
        }
        if self.parameters.get("VALUE2").is_none() {
            return Err("The Simple parameter set for 'VALUE2' has not been set.".into())
        }
        let mut constraints: Vec<Constraint> = Vec::new();
        fn multiple_of_x(v: &[usize]) -> bool { v[1] % v[0] == 0 };
        constraints.push(Constraint{func: multiple_of_x,
            args: vec!["VALUE1".into(), "VALUE2".into()]});
        Ok(ParameterSet{parameters: self.parameters, constraints: constraints})
    }
}
