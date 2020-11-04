use crate::layer::chain::LayerChain;
use crate::loss::loss::Loss;
use crate::traits::model_traits::Model;
use crate::traits::numerical_traits::{MLPFLoatRandSampling, MLPFloat};
use crate::traits::optimizer_traits::Optimizer;
use crate::traits::tensor_traits::Tensor;
use crate::utility::counter::CounterEst;
use ndarray::{ArrayD, ArrayViewD};

pub struct Serial<'a, T>
where
    T: MLPFloat,
{
    layer_chain: LayerChain<T>,
    loss_function: Loss<'a, T>,
}

impl<'a, T> Serial<'a, T>
where
    T: MLPFLoatRandSampling,
{
    pub fn new(loss_function: Loss<'a, T>) -> Self {
        Self {
            layer_chain: LayerChain::new(),
            loss_function,
        }
    }

    pub fn new_from_layers<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(
        layers: I,
        loss_function: Loss<'a, T>,
    ) -> Self {
        Self {
            layer_chain: LayerChain::new_from_sublayers(layers),
            loss_function,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Tensor<T>>) {
        self.layer_chain.push(layer);
    }

    pub fn add_all<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(&mut self, layers: I) {
        self.layer_chain.push_all(layers)
    }

    pub fn num_param(&self) -> CounterEst<usize> {
        self.layer_chain.num_parameters()
    }

    pub fn num_operations_per_forward(&self) -> CounterEst<usize> {
        self.layer_chain.num_operations_per_forward()
    }
}

impl<'a, T> Model<T> for Serial<'a, T>
where
    T: MLPFloat,
{
    fn train(
        &mut self,
        max_num_iter: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
    ) {
        // TODD: clones below are temp var for testing.
        let input_clone = input.clone();
        let expected_output_clone = expected_output.clone();
        for i in 0..max_num_iter {
            let forward_res = self.layer_chain.forward(input_clone.view());
            assert_eq!(forward_res.shape(), expected_output_clone.shape());
            let gradient = self.loss_function.backward_with_respect_to_input(
                forward_res.view(),
                expected_output_clone.view(),
                true,
            );
            println!(
                "Iter {}: loss={}",
                i,
                self.loss_function.calculate_loss(
                    forward_res.view(),
                    expected_output_clone.view(),
                    true
                )
            );
            self.layer_chain.backward_update_check_frozen(
                input_clone.view(),
                gradient.view(),
                optimizer,
            );
        }
    }

    fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.loss_function
            .predict(self.layer_chain.predict(input).view(), false)
    }

    fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.loss_function
            .predict(self.layer_chain.par_predict(input.into_dyn()).view(), true)
    }
}
