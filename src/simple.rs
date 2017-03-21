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

    pub fn build<'a>(self) -> Result<ParameterSet<'a>, String> {
        let ordered = vec!["VALUE1", "VALUE2"];
        for &name in &ordered{
            if self.parameters.get(name).is_none() {
                return Err(format!("The Simple parameter set for '{}' has not been set.", name))
            }
        }
        let parameters = ordered.iter().map(move |&x| {
            let s: String = x.into();
            let v = self.parameters[&s].clone();
            (s, v)
        }).collect();
        let mut constraints: Vec<Constraint<'static>> = Vec::new();
        fn multiple_of_x(v: &[usize]) -> bool { v[1] % v[0] == 0 };
        constraints.push(Constraint{func: multiple_of_x,
            args: vec!["VALUE1", "VALUE2"]});
        Ok(ParameterSet{parameters: parameters, constraints: constraints})
    }
}
