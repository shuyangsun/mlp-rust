use crate::layer::chain::LayerChain;
use crate::traits::numerical_traits::MLPFLoatRandSampling;
use ndarray::{Array1, ArrayView2, Ix2};

pub struct MLPClassifier<T>
where
    T: MLPFLoatRandSampling,
{
    layers: LayerChain<T>,
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

    pub fn predict(&self, input: ArrayView2<T>) -> Array1<T> {
        self.layers
            .predict(input.into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap()
            .into_shape([input.nrows()])
            .unwrap()
    }

    pub fn par_predict(&self, input: ArrayView2<T>) -> Array1<T> {
        self.layers
            .par_predict(input.into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap()
            .into_shape([input.nrows()])
            .unwrap()
    }
}
