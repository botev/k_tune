use core::{KernelWrapper, ParameterSet};
use std::collections::HashMap;

pub fn build_kernel_wrapper(m: usize, n: usize, k: usize) -> KernelWrapper {
    KernelWrapper {
        scalar_inputs: vec![m, n, k],
        inputs_dims: vec![(m, k), (k, n), (m, n)],
        name: "fast_gemm".into(),
        src: "gemm.ocl".into(),
        reference_src: None
    }
}

#[derive(Clone, Debug)]
pub struct GemmBuilder {
    parameters: HashMap<String, Vec<usize>>,
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

    pub fn mwg(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("MWG".into(), values);
        return self
    }

    pub fn nwg(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("NWG".into(), values);
        return self
    }

    pub fn kwg(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("KWG".into(), values);
        return self
    }

    pub fn mdimc(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("MDIMC".into(), values);
        return self
    }

    pub fn ndimc(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("NDIMC".into(), values);
        return self
    }

    pub fn mdima(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("MDIMA".into(), values);
        return self
    }

    pub fn ndimb(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("NDIMB".into(), values);
        return self
    }

    pub fn kwi(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("KWI".into(), values);
        return self
    }

    pub fn vwm(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("VWM".into(), values);
        return self
    }

    pub fn vwn(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("VWN".into(), values);
        return self
    }

    pub fn strm(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as usize).collect();
        self.parameters.insert("STRM".into(), values);
        return self
    }

    pub fn strn(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as usize).collect();
        self.parameters.insert("STRN".into(), values);
        return self
    }

    pub fn sa(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as usize).collect();
        self.parameters.insert("SA".into(), values);
        return self
    }

    pub fn sb(mut self, values: Vec<bool>) -> Self {
        let values = values.into_iter().map(|x| x as usize).collect();
        self.parameters.insert("SB".into(), values);
        return self
    }

    pub fn precision(mut self, values: Vec<usize>) -> Self {
        self.parameters.insert("PRECISION".into(), values);
        return self
    }

    pub fn build(self) -> Result<ParameterSet, String> {
        for name in vec!["MWG", "NWG", "KWG",
                         "MDIMC", "NDIMC", "MDIMA", "NDIMB",
                         "KWI", "VWM", "VWN",
                         "STRM", "STRN",
                         "SA", "SB", "PRECISION"] {
            if self.parameters.get(name).is_none() {
                return Err(format!("The GEMM parameter set for '{}' has not been set.", name))
            }
        }
        Ok(ParameterSet{parameters: self.parameters})
    }
}
