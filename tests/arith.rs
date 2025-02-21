use rusten::Tensor;

#[test]
fn test_broadcast_unary_ops() {
    let ops = vec![
        Tensor::abs,
        Tensor::neg,
        Tensor::ceil,
        Tensor::floor,
        Tensor::round,
        Tensor::exp,
        Tensor::exp2,
        Tensor::ln,
        Tensor::log10,
        Tensor::log2,
        Tensor::sin,
        Tensor::cos,
        Tensor::tan,
        Tensor::sinh,
        Tensor::cosh,
        Tensor::tanh,
        Tensor::sqrt,
    ];

    let tensor = Tensor::randn(vec![10], 0).abs().broadcast(vec![10, 10]);
    for op in ops {
        assert_eq!(op(&tensor.clone().contig()), op(&tensor));
    }
}

#[test]
fn test_broadcast_binary_ops() {
    let ops = vec![
        Tensor::add,
        Tensor::div,
        Tensor::mul,
        Tensor::sub,
        Tensor::maximum,
        Tensor::minimum,
        Tensor::pow,
    ];

    let tensor1 = Tensor::randn(vec![10], 0).abs().broadcast(vec![10, 10, 10]);
    let tensor2 = Tensor::randn(vec![10, 10], 1).abs();
    let tensor1_contig = tensor1.clone().contig();

    for op in ops {
        assert_eq!(op(&tensor1_contig, &tensor2), op(&tensor1, &tensor2));
    }

    let ops = vec![
        Tensor::eq,
        Tensor::ne,
        Tensor::gt,
        Tensor::lt,
        Tensor::ge,
        Tensor::le,
    ];

    for op in ops {
        assert_eq!(op(&tensor1_contig, &tensor2.clone().contig()), op(&tensor1, &tensor2));
    }
}

#[test]
fn test_broadcast_reduction_ops() {
    let ops = vec![
        Tensor::sum,
        Tensor::prod,
        Tensor::max,
        Tensor::min,
        Tensor::mean,
        Tensor::median,
        Tensor::var,
    ];
    let shapes = vec![
        (vec![10], vec![10], 0),
        (vec![10], vec![10, 10], 0),
        (vec![10], vec![10, 10], 1),
        (vec![10, 1], vec![10, 10], 0),
        (vec![10, 1], vec![10, 10], 1),
    ];

    for (shape, shape_broadcasted, axis) in shapes {
        let tensor = Tensor::randn(shape, 0).broadcast(shape_broadcasted);
        let tensor_contig = tensor.clone().contig();
        for op in ops.clone() {
            assert_eq!(op(&tensor_contig, axis), op(&tensor, axis));
        }
    }
}

#[test]
fn test_randn() {
    for seed in 0..32 {
        let tensor = Tensor::randn(vec![1000000], seed);
        assert!((tensor.mean(0).data()[0]).abs() < 0.01);
        assert!((tensor.var(0).data()[0] - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_permute_matmul() {
    let shapes = vec![
        (vec![10, 20], vec![10, 20], vec![1, 0], vec![0, 1]),
        (vec![10, 20], vec![10, 20], vec![0, 1], vec![1, 0]),
        (vec![10, 20], vec![20, 10], vec![1, 0], vec![1, 0]),
    ];

    for (shape1, shape2, ord1, ord2) in shapes {
        let tensor1 = Tensor::randn(shape1, 0).permute(&ord1);
        let tensor2 = Tensor::randn(shape2, 0).permute(&ord2);

        let tensor1_contig = tensor1.clone().contig();
        let tensor2_contig = tensor2.clone().contig();

        assert_eq!(tensor1_contig.matmul(&tensor2_contig), tensor1.matmul(&tensor2));
    }
}

#[test]
fn test_broadcast_matmul() {
    let shapes = vec![
        (vec![10, 1], vec![10], vec![10, 20], vec![20, 10]),
        (vec![10], vec![10, 1], vec![20, 10], vec![10, 20]),
    ];

    for (shape1, shape2, shape_broadcasted1, shape_broadcasted2) in shapes {
        let tensor1 = Tensor::randn(shape1, 0).broadcast(shape_broadcasted1);
        let tensor2 = Tensor::randn(shape2, 0).broadcast(shape_broadcasted2);

        let tensor1_contig = tensor1.clone().contig();
        let tensor2_contig = tensor2.clone().contig();

        assert_eq!(tensor1_contig.matmul(&tensor2_contig), tensor1.matmul(&tensor2));
    }
}
