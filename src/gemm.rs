use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

use core::*;
use ocl::SpatialDims;

pub fn build_kernel_wrapper(file: &str, m: usize, n: usize, k: usize) -> KernelWrapper {
    let mut src = String::new();
    File::open(file).unwrap().read_to_string(&mut src).unwrap();
    KernelWrapper {
        scalar_inputs: vec![m as i32, n as i32, k as i32],
        inputs_dims: vec![(m, k), (k, n), (m, n)],
        src: src,
        name: "gemm_fast".into(),
        ref_name: None,
        global_base: SpatialDims::Two(m, n),
        local_base: SpatialDims::Two(1, 1)
    }
}

#[derive(Clone, Debug)]
pub struct GemmBuilder {
    parameters: HashMap<String, Vec<i32>>,
}

impl Default for GemmBuilder {
    fn default() -> Self {
        GemmBuilder::new()
            .mwg(vec![64])
            .nwg(vec![64])
            .kwg(vec![8])
            .mdimc(vec![8])
            .ndimc(vec![8])
            .mdima(vec![8])
            .ndimb(vec![8])
            .kwi(vec![8])
            .vwm(vec![1])
            .vwn(vec![1])
            .strm(vec![true])
            .strn(vec![true])
            .sa(vec![true])
            .sb(vec![true])
            .precision(vec![32])
    }
}


impl GemmBuilder {
    pub fn new() -> Self {
        GemmBuilder{parameters: HashMap::new()}
    }

    pub fn mwg(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("MWG".into(), values);
        return self
    }

    pub fn nwg(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("NWG".into(), values);
        return self
    }

    pub fn kwg(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("KWG".into(), values);
        return self
    }

    pub fn mdimc(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("MDIMC".into(), values);
        return self
    }

    pub fn ndimc(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("NDIMC".into(), values);
        return self
    }

    pub fn mdima(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("MDIMA".into(), values);
        return self
    }

    pub fn ndimb(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("NDIMB".into(), values);
        return self
    }

    pub fn kwi(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("KWI".into(), values);
        return self
    }

    pub fn vwm(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("VWM".into(), values);
        return self
    }

    pub fn vwn(mut self, values: Vec<i32>) -> Self {
        self.parameters.insert("VWN".into(), values);
        return self
    }

    pub fn strm(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as i32).collect();
        self.parameters.insert("STRM".into(), values);
        return self
    }

    pub fn strn(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as i32).collect();
        self.parameters.insert("STRN".into(), values);
        return self
    }

    pub fn sa(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as i32).collect();
        self.parameters.insert("SA".into(), values);
        return self
    }

    pub fn sb(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as i32).collect();
        self.parameters.insert("SB".into(), values);
        return self
    }

    pub fn precision(mut self, values: Vec<i32>) -> Self {
        for &v in &values {
            if v != 32 && v != 64 {
                panic!("Precision can be only 32 or 64.")
            }
        }
        self.parameters.insert("PRECISION".into(), values);
        return self
    }

    pub fn build<'a>(self) -> Result<ParameterSet<'a>, String> {
        let ordered = vec!["MWG", "NWG", "KWG",
                           "MDIMC", "NDIMC", "MDIMA", "NDIMB",
                           "KWI", "VWM", "VWN",
                           "STRM", "STRN",
                           "SA", "SB", "PRECISION"];
        for &name in &ordered{
            if self.parameters.get(name).is_none() {
                return Err(format!("The GEMM parameter set for '{}' has not been set.", name))
            }
        }
        let parameters = ordered.iter().map(move |&x| {
            let s: String = x.into();
            let v = self.parameters[&s].clone();
            (s, v)
        }).collect();
        let mut constraints: Vec<Constraint<'static>> = Vec::new();
        fn multiple_of_x(v: &[i32]) -> bool { v[0] % v[1] == 0 }
        fn multiple_of_x_mul_y(v: &[i32]) -> bool { v[0] % (v[1] * v[2]) == 0 }
        fn multiple_of_x_mul_y_div_z(v: &[i32]) -> bool { v[0] % ((v[1] * v[2]) / v[3]) == 0 }

        // Sets constraints: Requirement for unrolling the KWG loop
        constraints.push(Constraint{func: multiple_of_x, args: vec!["KWG", "KWI"]});

        // Sets constraints: Required for integer MWI and NWI
        constraints.push(Constraint{func: multiple_of_x_mul_y, args: vec!["MWG", "MDIMC", "VWM"]});
        constraints.push(Constraint{func: multiple_of_x_mul_y, args: vec!["NWG", "NDIMC", "VWN"]});

        // Sets constraints: Required for integer MWIA and NWIB
        constraints.push(Constraint{func: multiple_of_x_mul_y, args: vec!["MWG", "MDIMA", "VWM"]});
        constraints.push(Constraint{func: multiple_of_x_mul_y, args: vec!["NWG", "NDIMB", "VWN"]});

        // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
        constraints.push(Constraint{func: multiple_of_x_mul_y_div_z,
            args: vec!["KWG", "MDIMC", "NDIMC", "MDIMA"]});
        constraints.push(Constraint{func: multiple_of_x_mul_y_div_z,
            args: vec!["KWG", "MDIMC", "NDIMC", "NDIMB"]});

        Ok(ParameterSet{
            parameters: parameters,
            constraints: constraints,
            mul_global_size: Some(vec![Some("MDIMC".into()), Some("NDIMC".into())]),
            mul_local_size: Some(vec![Some("MDIMC".into()), Some("NDIMC".into())]),
            div_global_size: Some(vec![Some("MWG".into()), Some("NWG".into())])
        })
    }
}
