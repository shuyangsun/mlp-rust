use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

const SMALL_SAMPLE_SIZES: [usize; 14] = [
    1, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
];

const LARGE_SAMPLE_SIZES: [usize; 11] = [
    1, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000,
];

const EXTRA_LARGE_SAMPLE_SIZES: [usize; 5] = [1, 250_000, 500_000, 750_000, 1_000_000];

const DNN_SMALL_LAYER_SIZE: [usize; 2] = [16, 4];
const DNN_LARGE_LAYER_SIZE: [usize; 10] = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8];

fn gen_data_set<T>(
    sample_size: usize,
    feature_size: usize,
    output_size: usize,
    test_data_ratio: f64,
) -> Box<dyn DataSet<T, IxDyn>>
where
    T: MLPFLoatRandSampling,
{
    let data = Array::random(
        (sample_size, feature_size + output_size),
        Uniform::new(T::zero(), T::one()),
    )
    .into_dyn();
    Box::new(DataSetInMemory::new(
        data,
        output_size,
        test_data_ratio,
        true,
    ))
}

fn mlp_forward_benchmark_small_network_small_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("SS");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in SMALL_SAMPLE_SIZES.iter() {
        let hidden_layer_sizes = Vec::from(DNN_SMALL_LAYER_SIZE);
        let simple_dnn_f32 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes.clone(),
            Activation::ReLu,
            false,
            false,
        );
        let dataset_f32 = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Ser32", size), size, |b, _| {
            b.iter(|| simple_dnn_f32.predict(dataset_f32.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Par32", size), size, |b, _| {
            b.iter(|| simple_dnn_f32.par_predict(dataset_f32.train_data().0))
        });

        let simple_dnn_f64 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::ReLu,
            false,
            false,
        );
        let dataset_f64 = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Ser64", size), size, |b, _| {
            b.iter(|| simple_dnn_f64.predict(dataset_f64.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Par64", size), size, |b, _| {
            b.iter(|| simple_dnn_f64.par_predict(dataset_f64.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_small_network_large_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("SL");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in LARGE_SAMPLE_SIZES.iter() {
        let hidden_layer_sizes = Vec::from(DNN_SMALL_LAYER_SIZE);

        let simple_dnn_f32 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes.clone(),
            Activation::ReLu,
            false,
            false,
        );
        let dataset_f32 = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Ser32", size), size, |b, _| {
            b.iter(|| simple_dnn_f32.predict(dataset_f32.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Par32", size), size, |b, _| {
            b.iter(|| simple_dnn_f32.par_predict(dataset_f32.train_data().0))
        });

        let simple_dnn_f64 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::ReLu,
            false,
            false,
        );
        let dataset_f64 = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Ser64", size), size, |b, _| {
            b.iter(|| simple_dnn_f64.predict(dataset_f64.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Par64", size), size, |b, _| {
            b.iter(|| simple_dnn_f64.par_predict(dataset_f64.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_large_network_small_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("LS");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in SMALL_SAMPLE_SIZES.iter() {
        let hidden_layer_sizes = Vec::from(DNN_LARGE_LAYER_SIZE);

        let simple_dnn_32 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes.clone(),
            Activation::TanH,
            false,
            false,
        );
        let dataset_32 = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Ser32", size), size, |b, _| {
            b.iter(|| simple_dnn_32.predict(dataset_32.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Par32", size), size, |b, _| {
            b.iter(|| simple_dnn_32.par_predict(dataset_32.train_data().0))
        });

        let simple_dnn_64 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::TanH,
            false,
            false,
        );
        let dataset_64 = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Ser64", size), size, |b, _| {
            b.iter(|| simple_dnn_64.predict(dataset_64.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Par64", size), size, |b, _| {
            b.iter(|| simple_dnn_64.par_predict(dataset_64.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_large_network_large_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("LL");
    group.sample_size(10);
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in LARGE_SAMPLE_SIZES.iter() {
        let hidden_layer_sizes = Vec::from(DNN_LARGE_LAYER_SIZE);

        let simple_dnn_32 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes.clone(),
            Activation::TanH,
            false,
            false,
        );
        let dataset_32 = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Par32", size), size, |b, _| {
            b.iter(|| simple_dnn_32.par_predict(dataset_32.train_data().0))
        });

        let simple_dnn_64 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::TanH,
            false,
            false,
        );
        let dataset_64 = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Par64", size), size, |b, _| {
            b.iter(|| simple_dnn_64.par_predict(dataset_64.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_large_network_extra_large_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("LXL");
    group.sample_size(10);
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in EXTRA_LARGE_SAMPLE_SIZES.iter() {
        let hidden_layer_sizes = Vec::from(DNN_LARGE_LAYER_SIZE);

        let simple_dnn_32 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes.clone(),
            Activation::TanH,
            false,
            false,
        );
        let dataset_32 = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Par32", size), size, |b, _| {
            b.iter(|| simple_dnn_32.par_predict(dataset_32.train_data().0))
        });

        let simple_dnn_64 = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::TanH,
            false,
            false,
        );
        let dataset_64 = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        group.bench_with_input(BenchmarkId::new("Par64", size), size, |b, _| {
            b.iter(|| simple_dnn_64.par_predict(dataset_64.train_data().0))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    mlp_forward_benchmark_small_network_small_sample,
    mlp_forward_benchmark_small_network_large_sample,
    mlp_forward_benchmark_large_network_small_sample,
    mlp_forward_benchmark_large_network_large_sample,
    mlp_forward_benchmark_large_network_extra_large_sample
);
criterion_main!(benches);
