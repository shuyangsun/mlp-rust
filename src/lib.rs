mod data_set;
mod layer;
mod loss;
mod model;
mod optimizer;
pub mod prelude;
mod traits;
mod utility;

pub use self::data_set::in_memory::DataSet2D;
pub use self::layer::{
    activation::Activation, batch_normalization::BatchNormalization, bias::Bias, chain::LayerChain,
    dense::Dense, input_normalization::InputNormalization,
};
pub use self::loss::loss::Loss;
pub use self::model::{mlp::MLP, serial::Serial};
pub use self::optimizer::gradient_descent::GradientDescent;
pub use self::traits::{
    data_set_traits::DataSet, data_set_traits::InputOutputData, model_traits::Model,
    numerical_traits::MLPFLoatRandSampling, numerical_traits::MLPFloat,
    optimizer_traits::Optimizer, tensor_traits::Tensor,
};
