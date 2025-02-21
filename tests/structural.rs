use rusten::Tensor;

#[test]
fn test_broadcast_permute_reshape() {
    let shapes = vec![
        vec![1],
        vec![1, 1],
        vec![1, 1, 1],
        vec![4],
        vec![3, 1],
        vec![3, 4],
        vec![2, 1, 1],
        vec![2, 1, 4],
        vec![2, 3, 1],
        vec![2, 3, 4],
    ];

    for shape in shapes {
        let tensor1 = Tensor::randn(shape, 0).broadcast(vec![2, 3, 4]);

        let tensor2 = tensor1.clone()
            .permute(&[1, 2, 0])
            .reshape(vec![12, 2])
            .permute(&[1, 0])
            .reshape(vec![2, 3, 4]);
        assert_eq!(tensor1, tensor2);

        let tensor2 = tensor1.clone()
        .reshape(vec![2, 12])
        .permute(&[1, 0])
        .reshape(vec![3, 4, 2])
        .permute(&[2, 0, 1]);
        assert_eq!(tensor1, tensor2);
    }
}

#[test]
fn test_permute_concat() {
    let shapes = vec![
        (vec![1, 4], vec![2, 4], vec![1, 0], 1),
        (vec![2, 3], vec![2, 4], vec![1, 0], 0),
        (vec![2, 3, 4], vec![5, 3, 4], vec![0, 2, 1], 0),
        (vec![2, 3, 4], vec![5, 3, 4], vec![1, 2, 0], 2),
        (vec![2, 3, 4], vec![2, 5, 4], vec![2, 1, 0], 1),
        (vec![2, 3, 4], vec![2, 5, 4], vec![1, 2, 0], 0),
        (vec![2, 3, 4], vec![2, 3, 5], vec![1, 0, 2], 2),
        (vec![2, 3, 4], vec![2, 3, 5], vec![1, 2, 0], 1),
    ];

    for (shape1, shape2, ord, axis) in shapes {
        let len1: usize = shape1.iter().product();
        let len2: usize = shape2.iter().product();

        let tensor1 = Tensor::new((1..=len1).collect(), shape1);
        let tensor2 = Tensor::new((len1 + 1..=len1 + len2).collect(), shape2);

        let exp = tensor1.concat(&tensor2, ord[axis]).permute(&ord);
        let concat = (tensor1.permute(&ord)).concat(&tensor2.permute(&ord), axis);

        assert_eq!(exp, concat);
    }
}

#[test]
fn test_permute_slice() {
    let tensor = Tensor::new((1..=18).collect(), vec![2, 3, 3])
        .permute(&[1, 2, 0]);
    let start = vec![0, 2, 0];
    let end = vec![2, 3, 2];
    let sliced = tensor.slice(&start, &end);

    assert_eq!(Tensor::new(vec![3, 12, 6, 15], vec![2, 1, 2]), sliced);
}

#[test]
fn test_broadcast_slice() {
    let tensor = Tensor::new(vec![1, 2, 3], vec![3])
        .broadcast(vec![2, 3, 3]);
    let start = vec![0, 2, 0];
    let end = vec![2, 3, 2];
    let sliced = tensor.slice(&start, &end);

    assert_eq!(Tensor::new(vec![1, 2, 1, 2], vec![2, 1, 2]), sliced);
}
