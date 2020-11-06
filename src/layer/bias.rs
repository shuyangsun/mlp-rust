use crate::utility::{array::add_1_dim_to_shape, counter::CounterEst};
use crate::{MLPFloat, Optimizer, Tensor};
use ndarray::{Array, ArrayView, Dimension, ShapeBuilder};

pub struct Bias<T, D>
where
    T: MLPFloat,
{
    is_frozen: bool,
    bias_arr: Array<T, D>,
}

impl<T, D> Tensor<T, D, D> for Bias<T, D>
where
    D: 'static,
    T: MLPFloat,
{
    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn forward(&self, input: ArrayView<T, D>) -> Array<T, D> {
        let bias_arr_broadcasted_view = self.bias_arr.broadcast(input.dim()).unwrap();
        &input + &bias_arr_broadcasted_view
    }

    fn backward_respect_to_input(
        &self,
        _: ArrayView<T, D>,
        layer_output: ArrayView<T, D>,
    ) -> Array<T, D> {
        layer_output.into_owned()
    }

    fn backward_update(
        &mut self,
        _: ArrayView<T, D>,
        output_gradient: ArrayView<T, D>,
        optimizer: &Box<dyn Optimizer<T, D>>,
    ) {
        optimizer.change_values(&mut self.bias_arr.view_mut(), output_gradient);
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.bias_arr.len())
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.bias_arr.len())
    }
}

impl<T, D> Bias<T, D>
where
    T: MLPFloat,
{
    pub fn new<Sh>(shape: Sh) -> Self
    where
        D: Dimension,
        Sh: ShapeBuilder<Dim = D::Smaller>,
    {
        let new_shape = add_1_dim_to_shape(shape);
        assert_eq!(new_shape.size(), shape.size());
        Self {
            is_frozen: false,
            bias_arr: Array::zeros(new_shape),
        }
    }

    pub fn new_frozen<Sh>(shape: Sh) -> Self
    where
        D: Dimension,
        Sh: ShapeBuilder<Dim = D::Smaller>,
    {
        let mut res = Self::new(shape);
        res.is_frozen = true;
        res
    }
}

#[macro_export]
macro_rules! bias {
    ($x:expr) => {{
        Box::new(Bias::new($x))
    }};
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;

    use super::super::super::traits::tensor_traits::Tensor;
    use super::Bias;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_bias_forward() {
        let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
        let bias = Bias::new(2);
        let forward_res = bias.forward(arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(&forward_res, arr);
    }

    #[test]
    fn test_bias_forward_rand() {
        let shape = [1000, 100];
        let rand_arr = &Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let weights = Bias::new(100);
        let forward_res = weights.forward(rand_arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(&forward_res, rand_arr);
    }
}
