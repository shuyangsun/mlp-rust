use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

const SMALL_SAMPLE_SIZES: [usize; 17] = [
    1, 10, 20, 50, 100, 200, 500, 700, 1000, 2000, 3000, 5000, 6_000, 7_000, 8_000, 9_000, 10_000,
];

const LARGE_SAMPLE_SIZES: [usize; 12] = [
    1, 20_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 650_000, 800_000, 900_000,
    1_000_000,
];

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

fn mlp_forward_benchmark_small_network_small_sample_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("SS32");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in SMALL_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_SMALL_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::ReLu,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Serial", size), size, |b, _| {
            b.iter(|| simple_dnn.predict(dataset.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_small_network_large_sample_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("SL32");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in LARGE_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_SMALL_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::ReLu,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Serial", size), size, |b, _| {
            b.iter(|| simple_dnn.predict(dataset.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_small_network_small_sample_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("SS64");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in SMALL_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_SMALL_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::ReLu,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Serial", size), size, |b, _| {
            b.iter(|| simple_dnn.predict(dataset.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_small_network_large_sample_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("SL64");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in LARGE_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_SMALL_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::ReLu,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Serial", size), size, |b, _| {
            b.iter(|| simple_dnn.predict(dataset.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_large_network_small_sample_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("LS32");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in SMALL_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_LARGE_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::TanH,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Serial", size), size, |b, _| {
            b.iter(|| simple_dnn.predict(dataset.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_large_network_large_sample_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("LL32");
    group.sample_size(10);
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in LARGE_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f32>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_LARGE_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::TanH,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_large_network_small_sample_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("LS64");
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in SMALL_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_LARGE_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::TanH,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Serial", size), size, |b, _| {
            b.iter(|| simple_dnn.predict(dataset.train_data().0))
        });
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

fn mlp_forward_benchmark_large_network_large_sample_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("LL64");
    group.sample_size(10);
    let feature_size = 28 * 28;
    let output_size = 1;

    for size in LARGE_SAMPLE_SIZES.iter() {
        let dataset = gen_data_set::<f64>(size.clone(), feature_size, output_size, 0.);
        let hidden_layer_sizes = Vec::from(DNN_LARGE_LAYER_SIZE);
        let simple_dnn = MLP::new_regressor(
            feature_size,
            output_size,
            hidden_layer_sizes,
            Activation::TanH,
            false,
            false,
        );
        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, _| {
            b.iter(|| simple_dnn.par_predict(dataset.train_data().0))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    mlp_forward_benchmark_small_network_small_sample_f32,
    mlp_forward_benchmark_small_network_large_sample_f32,
    mlp_forward_benchmark_small_network_small_sample_f64,
    mlp_forward_benchmark_small_network_large_sample_f64,
    mlp_forward_benchmark_large_network_small_sample_f32,
    mlp_forward_benchmark_large_network_large_sample_f32,
    mlp_forward_benchmark_large_network_small_sample_f64,
    mlp_forward_benchmark_large_network_large_sample_f64,
);
criterion_main!(benches);
