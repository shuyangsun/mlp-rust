use crate::layer::chain::LayerChain;
use crate::traits::numerical_traits::MLPFLoatRandSampling;
use crate::traits::tensor_traits::{Tensor, TensorTraitObjWrapper};
use crate::utility::counter::CounterEst;
use ndarray::{Array1, ArrayView2, Ix2};

pub struct Model<T>
where
    T: MLPFLoatRandSampling,
{
    layer_chain: LayerChain<T>,
}

impl<T> Model<T>
where
    T: MLPFLoatRandSampling,
{
    pub fn new() -> Self {
        Self {
            layer_chain: LayerChain::new(),
        }
    }

    pub fn train(input: ArrayView2<T>, expected_output: ArrayView2<T>) {
        unimplemented!()
    }

    pub fn add(&mut self, layer: TensorTraitObjWrapper<T>) {
        self.layer_chain.push(layer);
    }

    pub fn add_all<I: IntoIterator<Item = TensorTraitObjWrapper<T>>>(&mut self, layers: I) {
        self.layer_chain.push_all(layers)
    }

    pub fn num_param(&self) -> CounterEst<usize> {
        self.layer_chain.num_parameters()
    }

    pub fn num_operations_per_forward(&self) -> CounterEst<usize> {
        self.layer_chain.num_operations_per_forward()
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
