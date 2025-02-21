# rusten

• **[Introduction](#introduction)**<br>
• **[Example](#example)**<br>
• **[Installation](#installation)**<br>
• **[Documentation](#documentation)**<br>
• **[Tests](#tests)**<br>

## Introduction

rusten is a minimal, didactic implementation of tensors, or, more accurately, multi-dimensional arrays, in Rust. It mimics [NumPy](https://numpy.org/) in terms of style and offers core functionalities such as unary, binary, and reduction operations, broadcasting, shape transformations, slicing, and more, all while prioritizing clarity. Unlike existing packages like [ndarray](https://github.com/rust-ndarray/ndarray), rusten doesn't aim for state-of-the-art performance or support for all use cases. Instead, it's targeted towards those seeking to gain a bare-bones understanding of multi-dimensional containers and related operations, without emphasizing performance or a comprehensive feature set. Coupled with Rust's intuitive syntax and clean memory management model, this means the rusten codebase is easy to explore and extend. It's also worth mentioning that, despite its compact codebase, rusten covers a surprisingly broad range of applications; in fact, with the major exception of convolutional networks, it can express the forward passes of most deep learning models, including transformers.

The motivation behind rusten is that truly understanding how tensor manipulation works under the hood helps fluency in tools like PyTorch. Although there are myriad from-scratch projects devoted to other aspects of deep learning pipelines - [common architectures from scratch](https://github.com/rasbt/deeplearning-models), [machine learning algorithms from scratch](https://github.com/eriklindernoren/ML-From-Scratch), [autodiff from scratch](https://github.com/karpathy/micrograd), ... - there don't seem to be any learner-oriented resources on how tensors are handled on a low level. A very rudimentary tensor structure is straightforward, but its complexity grows exponentially with the addition of modern features such as broadcasting, non-contiguous views, etc., and rusten's goal is to implement these processes as transparently and plainly as possible to aid interested students. The most similar project is [the Tensor](https://github.com/EurekaLabsAI/tensor) by [Eureka Labs](https://eurekalabs.ai/), but whereas that is more concerned with fine-grained C memory management and concentrates on one-dimensional vectors, rusten's focus is more on the aforementioned advanced array functionalities.

## Installation

rusten has no dependencies besides Rust itself, whose installation guide can be found [here](https://www.rust-lang.org/tools/install). To use it in your own projects, please add `rusten = { git = "https://github.com/bobmcdear/rusten.git", branch = "main" }` under the `[dependencies]` section of your `Cargo.toml` file.

## Example

The example below generates synthetic data constituting 128 data points and 100 features, and fits a linear regression model to it (no bias term) using gradient descent. It can be run by executing `cargo run --release` in the [`examples/lin_reg/`](https://github.com/BobMcDear/rusten/tree/main/examples/lin_reg) directory, where the `--release` flag instructs Cargo to build the program with optimizations.

```rust
use rusten::Tensor;

fn main() {
    let n = 128;
    let input = Tensor::randn(vec![n, 100], 0);
    let theta_true = Tensor::randn(vec![100, 1], 1);
    let targ = input.matmul(&theta_true);

    let lr = 0.5;
    let n_iters = 1000;
    let mut theta = Tensor::randn(vec![100, 1], 2);

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
```

As expected, the final difference will be negligible.

## Documentation

The rusten codebase is thoroughly annotated with doc comments, comprising examples for each method and an overview of its workings. Thanks to [rustdoc](https://doc.rust-lang.org/rustdoc/), these can be seamlessly converted into an actual documentation; simply clone this repository, navigate to it, and run `cargo doc --open`, as seen below:

```bash
git clone https://github.com/BobMcDear/rusten
cd rusten/
cargo doc --open
```

This opens up the documentation in your web browser.

## Tests

Besides the examples in the doc comments, rusten ships with tests to ensure it behaves as expected, particularly when composing together operations that might not interact well together. They can be invoked by running `cargo test` at the top-level of this repository.