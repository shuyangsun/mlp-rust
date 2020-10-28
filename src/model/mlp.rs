use crate::layer::chain::LayerChain;
use crate::traits::numerical_traits::MLPFLoatRandSampling;
use crate::traits::tensor_traits::Tensor;
use ndarray::{Array1, ArrayView2, Ix2};

pub struct MLPClassifier<T>
where
    T: MLPFLoatRandSampling,
{
    layer_chain: LayerChain<T>,
}

impl<T> MLPClassifier<T>
where
    T: MLPFLoatRandSampling,
{
    pub fn new(layer_sizes: &Vec<usize>) -> Self {
        unimplemented!()
    }

    pub fn train(input: ArrayView2<T>, output: ArrayView2<T>) {
        unimplemented!()
    }

    pub fn num_param(&self) -> Option<usize> {
        self.layer_chain.num_param()
    }

    pub fn predict(&self, input: ArrayView2<T>) -> Array1<T> {
        self.layer_chain
            .predict(input.into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap()
            .into_shape([input.nrows()])
            .unwrap()
    }

    pub fn par_predict(&self, input: ArrayView2<T>) -> Array1<T> {
        self.layer_chain
            .par_predict(input.into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap()
            .into_shape([input.nrows()])
            .unwrap()
    }
}
