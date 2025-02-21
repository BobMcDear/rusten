use rusten::Tensor;

fn main() {
    let n = 128;
    let d = 100;
    let input = Tensor::randn(vec![n, d], 0);
    let theta_true = Tensor::randn(vec![d, 1], 1);
    let targ = input.matmul(&theta_true);

    let lr = 0.5;
    let n_iters = 1000;
    let mut theta = Tensor::randn(vec![d, 1], 2);

    for _ in 0..n_iters {
        let grad = (&input.matmul(&theta) - &targ)
            .permute(&[1, 0])
            .matmul(&input)
            .permute(&[1, 0])
            / (n as f32).into();
        theta = theta - &lr.into() * &grad;
    }

    println!(
        "Sum of absolute difference between true and trained coefficients: {:?}",
        (theta - theta_true).abs().sum(0),
    );
}
