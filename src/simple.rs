use core::{KernelWrapper, ParameterSet};
use std::collections::HashMap;

pub fn build_kernel_wrapper(m: usize, n: usize) -> KernelWrapper {
    KernelWrapper {
        scalar_inputs: vec![],
        inputs_dims: vec![(m, n), (m, n), (m, n)],
        name: "add".into(),
        src: "simple.ocl".into(),
        reference_src: None
    }
}

#[derive(Clone, Debug)]
pub struct SimpleBuilder {
    parameters: HashMap<String, Vec<usize>>,
}

impl Default for SimpleBuilder {
    fn default() -> Self {
        SimpleBuilder::new().value(vec![1])
    }
}

impl SimpleBuilder {
    pub fn new() -> Self {
        SimpleBuilder{parameters: HashMap::new()}
    }

    pub fn value(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("VALUE".into(), values);
        return self
    }

    pub fn build(self) -> Result<ParameterSet, String> {
        if self.parameters.get("VALUE").is_none() {
            return Err("The Simple parameter set for 'VALUE' has not been set.".into())
        }
        Ok(ParameterSet{parameters: self.parameters})
    }
}
