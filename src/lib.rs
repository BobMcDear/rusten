use std::f32::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! unary_op {
    ($doc:literal, $name:ident, $op:expr) => {
        #[doc = $doc]
        pub fn $name(&self) -> Self {
            self.apply_uop($op)
        }
    };

    ($doc:literal, $name:ident, $op:expr, $out_ty:ty) => {
        #[doc = $doc]
        pub fn $name(&self) -> Tensor<$out_ty> {
            self.apply_uop($op)
        }
    };
}

macro_rules! binary_op {
    ($doc:literal, $name:ident, $op:expr) => {
        #[doc = $doc]
        pub fn $name(&self, other: &Self) -> Self {
            self.apply_bop(other, $op)
        }
    };

    ($doc:literal, $name:ident, $op:expr, $out_ty:ty) => {
        #[doc = $doc]
        pub fn $name(&self, other: &Self) -> Tensor<$out_ty> {
            self.apply_bop(other, $op)
        }
    };
}

macro_rules! reduction_op {
    ($doc:literal, $name:ident, $op:expr) => {
        #[doc = $doc]
        pub fn $name(&self, axis: usize) -> Self {
            self.apply_red(axis, $op)
        }
    };

    ($doc:literal, $name:ident, $op:expr, $out_ty:ty) => {
        #[doc = $doc]
        pub fn $name(&self, axis: usize) -> Tensor<$out_ty> {
            self.apply_red(axis, $op)
        }
    };
}

macro_rules! arith_trait {
    ($doc:literal, $name:ident, $method:ident) => {
        #[doc = $doc]
        impl<T> $name for Tensor<T>
        where
            T: Add<Output = T>
                + Div<Output = T>
                + Mul<Output = T>
                + Neg<Output = T>
                + Sub<Output = T>
                + PartialEq
                + PartialOrd
                + Default
                + Copy,
        {
            type Output = Tensor<T>;

            fn $method(self, other: Self) -> Self {
                Tensor::$method(&self, &other)
            }
        }

        #[doc = $doc]
        impl<T> $name for &Tensor<T>
        where
            T: Add<Output = T>
                + Div<Output = T>
                + Mul<Output = T>
                + Neg<Output = T>
                + Sub<Output = T>
                + PartialEq
                + PartialOrd
                + Default
                + Copy,
        {
            type Output = Tensor<T>;

            fn $method(self, other: Self) -> Tensor<T> {
                Tensor::$method(self, other)
            }
        }
    };
}

/// Tensor container (in actuality, a multi-dimensional array).
/// A tensor consists of the actual data, stored as a flat vector,
/// a shape specifying the dimensionality of each axis, and
/// strides, which are the number of elements to skip to reach the next
/// element along each axis.
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Implements basic tensor operations that are independent of the underlying type.
///
/// They're predominantly structural operations - that is, those that manipulate not the
/// data itself, but how it's arranged as a tensor (reshaping, broadcasting, etc.) - but there are
/// also three generic higher-order operators, namely, the unary, binary, and reduction operators.
impl<T> Tensor<T>
where
    T: Copy,
{
    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Computes the strides of a contiguous tensor of a given shape.
    ///
    /// Strides denote the number of elements to skip to reach the next element along each axis.
    /// For instance, consider a contiguous rank-3 tensor of shape `[4, 6, 10]`:
    /// To jump from index `[i, j, k]` to `[i+1, j, k]`, it's necessary to
    /// skip a subtensor (matrix) of shape `[6, 10]`, bringing the stride of the first axis to 60 = 6 * 10.
    /// Similarly, jumping from `[i, j, k]` to `[i, j+1, k]` requires
    /// skipping a subtensor (vector) of shape `[10]`, so the stride of the second axis is 10.
    /// Finally, the elements along the last axis are consecutive, meaning the stride is just 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let shape = vec![4, 6, 10];
    /// let strides = Tensor::<usize>::comp_strides(&shape);
    ///
    /// assert_eq!(vec![60, 10, 1], strides);
    /// ```
    pub fn comp_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Converts a vector of coordinates into a scalar index given strides.
    ///
    /// Coordinates are vectors comprising the index of the desired element along each axis.
    /// On the other hand, a scalar index specifies elements of the underlying flat vector of a tensor,
    /// i.e., it indexes the data as it's arranged in memory, not how it's interpreted using shapes and strides.
    /// For instance, consider a contiguous rank-3 tensor of shape `[4, 6, 10]`;
    /// this tensor is represented as a 1D vector with 240 = 4 * 6 * 10 in memory, with strides `[60, 10, 1]`.
    /// Since strides are the number of elements that must be skipped to jump to the next element along each axis,
    /// coordinate `[i, j, k]` corresponds to index 60 * i + 10 * j + 1 * k of the underlying vector.
    /// In other words, with 0 as the origin, 60 * i jumps i elements along the first axis,
    /// 10 * j jumps j elements along the second axis, and 1 * k jumps k elements along the third axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let coords = [2, 3, 4];
    /// let strides = [60, 10, 1];
    /// let idx = Tensor::<usize>::coords_to_idx(&coords, &strides);
    ///
    /// assert_eq!(60 * 2 + 10 * 3 + 4 * 1, idx);
    /// ```
    pub fn coords_to_idx(coords: &[usize], strides: &[usize]) -> usize {
        coords
            .iter()
            .zip(strides.iter())
            .map(|(coord, stride)| coord * stride)
            .sum()
    }

    /// Converts a scalar index into a vector of coordinates given strides.
    ///
    /// A scalar index specifies elements of the underlying flat vector of a tensor.
    /// On the other hand, coordinates are vectors comprising the index of the desired element along each axis.
    /// For instance, consider a contiguous rank-3 tensor of shape `[4, 6, 10]`;
    /// this tensor is represented as a 1D vector with 240 = 4 * 6 * 10 in memory, with strides `[60, 10, 1]`.
    /// Since strides are the number of elements that must be skipped to jump to the next element along each axis,
    /// index 60 * i + 10 * j + 1 * k of the underlying vector corresponds to the coordinate `[i, j, k]`.
    /// In other words, with 0 as the origin, 60 * i jumps i elements along the first axis,
    /// 10 * j jumps j elements along the second axis, and 1 * k jumps k elements along the third axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let idx = 60 * 2 + 10 * 3 + 4 * 1;
    /// let strides = [60, 10, 4];
    /// let coords = Tensor::<usize>::idx_to_coords(idx, &strides);
    ///
    /// assert_eq!(vec![2, 3, 1], coords);
    /// ```
    pub fn idx_to_coords(mut idx: usize, strides: &[usize]) -> Vec<usize> {
        let mut coords = Vec::with_capacity(strides.len());
        for &stride in strides.iter() {
            if stride == 0 {
                coords.push(0);
            } else {
                coords.push(idx / stride);
                idx %= stride;
            }
        }
        coords
    }

    /// Translates an index from one strides basis to another.
    ///
    /// Given the same data and shape, an index in memory could refer to two different elements
    /// depending on the stride. For example, consider the following matrix of shape `[2, 3]`:
    /// ```ignore
    /// 1 2 3
    /// 4 5 6
    /// ```
    /// Representing this matrix in memory in a row-major format (strides `[3, 1]`)
    /// yields `1 2 3 4 5 6`, whereas its column-major format (strides `[1, 3]`) would be `1 4 2 5 3 6`.
    /// Thus, index 1 (zero-based) refers to 2 in the case of the former and 4 in the case of the latter.
    /// This function translates an index from one strides basis to another, so that its output refers
    /// to the same element in the target tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let idx = 1;
    /// let strides_base = [3, 1];
    /// let strides_targ = [1, 3];
    /// let idx_translated = Tensor::<usize>::translate_idx(idx, &strides_base, &strides_targ);
    ///
    /// assert_eq!(3, idx_translated);
    /// ```
    pub fn translate_idx(idx: usize, strides_base: &[usize], strides_targ: &[usize]) -> usize {
        let coords = Self::idx_to_coords(idx, &strides_base);
        let idx_translated = Self::coords_to_idx(&coords, strides_targ);
        idx_translated
    }

    /// Finds the smallest shape both arguments can be broadcasted to.
    ///
    /// Starting from their right-most axes, two shapes are broadcasted
    /// according to the following rules. When one shape is shorter than the other,
    /// it's implicitly padded on the left with 1s.
    /// * If the dimensions match, the broadcasted shape is that.
    /// * If one is 1 and the other an arbitrary number, the broadcasted shape is the latter.
    /// * Otherwise, the two shapes can't be broadcasted to a common shape.
    ///
    /// # Panics
    ///
    /// This function panics if the shapes can't be broadcasted to a common shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let shape1 = [4, 1, 10];
    /// let shape2 = [6, 10];
    /// let shape_broadcasted = Tensor::<usize>::broadcast_shapes(&shape1, &shape2);
    ///
    /// assert_eq!(vec![4, 6, 10], shape_broadcasted);
    /// ```
    pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let len_broadcasted = len1.max(len2);
        let mut shape_broadcasted = Vec::with_capacity(len_broadcasted);

        for i in 0..len_broadcasted {
            let idx1 = len_broadcasted - len1;
            let idx2 = len_broadcasted - len2;
            let dim1 = if i < idx1 { 1 } else { shape1[i - idx1] };
            let dim2 = if i < idx2 { 1 } else { shape2[i - idx2] };

            if dim1 == dim2 {
                shape_broadcasted.push(dim1);
            } else if dim1 == 1 {
                shape_broadcasted.push(dim2);
            } else if dim2 == 1 {
                shape_broadcasted.push(dim1);
            } else {
                panic!("Shapes can't be broadcasted to a common shape.")
            }
        }

        shape_broadcasted
    }

    /// Calculates the strides corresponding to a shape after it's broadcasted.
    ///
    /// Setting the stride of an axis to 0 is equivalent to repeating it indefinitely
    /// across lower-order dimensions, effectively performing broadcasting.
    /// This function replaces a tensor's strides at axes that are to be broadcasted (i.e.,
    /// singleton axes and missing axes on the left) with 0s.
    ///
    /// # Panics
    ///
    /// This function panics if the original shape can't be broadcasted to the given shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let shape = [1, 10];
    /// let strides = [10, 1];
    /// let shape_broadcasted = [4, 6, 10];
    /// let strides_broadcasted = Tensor::<usize>::broadcast_strides(&strides, &shape, &shape_broadcasted);
    ///
    /// assert_eq!(vec![0, 0, 1], strides_broadcasted);
    /// ```
    pub fn broadcast_strides(
        strides: &[usize],
        shape: &[usize],
        shape_broadcasted: &[usize],
    ) -> Vec<usize> {
        let mut strides_padded = vec![shape.iter().product(); shape_broadcasted.len() - shape.len()];
        let mut shape_padded = vec![1; strides_padded.len()];
        strides_padded.extend(strides);
        shape_padded.extend(shape);

        let mut strides_broadcasted = Vec::with_capacity(shape_broadcasted.len());
        for (idx, (&dim_orig, &dim_broadcasted)) in
            shape_padded.iter().zip(shape_broadcasted).enumerate()
        {
            if dim_orig == dim_broadcasted {
                strides_broadcasted.push(strides_padded[idx]);
            } else if dim_orig == 1 {
                strides_broadcasted.push(0);
            } else {
                panic!("Original shape can't be broadcasted to the given shape");
            }
        }

        strides_broadcasted
    }

    /// Creates a new tensor of the given shape from 1D data.
    ///
    /// Tensors are represented as 1D data in memory; that is,
    /// their underlying format is shape- and layout-agnostic.
    /// Shape and stride are stored as metadata and used to interpret
    /// the flat data as a multi-dimensional array.
    ///
    /// # Panics
    ///
    /// This function panics if the data size and shape are incompatible.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![1, 2, 3], vec![3]);
    /// let tensor2 = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// ```
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Data size and shape don't match."
        );

        Self {
            data,
            strides: Self::comp_strides(&shape),
            shape,
        }
    }

    /// Returns true if the tensor is contiguous and false otherwise.
    ///
    /// A tensor is contiguous if and only if its elements are stored in memory
    /// as they would be traversed in a row-major order.
    /// This means the rule `strides[i] = strides[i + 1] * shape[i + 1]` must be satisfied
    /// for all `i` smaller than the rank of the tensor, and the last stride must be 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::randn(vec![4, 6, 10], 0);
    /// let tensor2 = Tensor::randn(vec![4, 6, 10], 0).permute(&[1, 2, 0]);
    ///
    /// assert!(tensor1.is_contig());
    /// assert!(!tensor2.is_contig());
    /// ```
    pub fn is_contig(&self) -> bool {
        Self::comp_strides(&self.shape) == self.strides
    }

    /// Makes the tensor contiguous.
    ///
    /// Because the row-major view of a non-contiguous tensor doesn't match its layout in memory,
    /// making it contiguous involves constructing a new tensor whose data in memory
    /// follows its row-major order. This entails allocating a new buffer and copying
    /// elements in a row-major order from the original tensor, as opposed to traversing
    /// the non-contiguous data as it's stored in memory.
    /// For instance, consider the following non-contiguous matrix of shape `[3, 2]` and strides `[1, 3]`,
    /// which is represented as the 1D vector `0 1 2 3 4 5` in memory.
    /// ```ignore
    /// 0 3
    /// 1 4
    /// 2 5
    /// ```
    /// The contiguous equivalent of this matrix would be stored in memory in a row-major order,
    /// i.e., `0 3 1 4 2 5`, and have the same shape of `[3, 2]` but strides of `[2, 1]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![4, 6, 10], 0)
    ///     .permute(&[1, 2, 0])
    ///     .contig();
    ///
    /// assert!(tensor.is_contig());
    /// ```
    pub fn contig(self) -> Self {
        if self.is_contig() {
            return self;
        }

        let n = self.shape.iter().product();
        let mut data_contig = Vec::with_capacity(n);
        let strides_contig = Self::comp_strides(&self.shape);

        for idx_contig in 0..n {
            let idx_orig = Self::translate_idx(idx_contig, &strides_contig, &self.strides);
            data_contig.push(self.data[idx_orig]);
        }

        Self::new(data_contig, self.shape)
    }

    /// Reshapes the tensor.
    ///
    /// Generally, a tensor can be reshaped by modifying its metadata, namely, shape and strides.
    /// However, non-contiguous tensors must first be made contiguous.
    ///
    /// # Panics
    ///
    /// This function panics if the given shape isn't compatible with the tensor's element count.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![4, 6, 10], 0).reshape(vec![4, 60]);
    ///
    /// assert_eq!(vec![4, 60], tensor.shape());
    /// assert_eq!(vec![60, 1], tensor.strides());
    /// ```
    pub fn reshape(self, shape: Vec<usize>) -> Self {
        assert_eq!(
            self.shape.iter().product::<usize>(),
            shape.iter().product(),
            "Original shape and given shape aren't compatible."
        );

        Self::new(self.contig().data, shape)
    }

    /// Permutes the tensor's axes.
    ///
    /// Whereas reshaping merges or splits a tensor's axes, permutations swaps them.
    /// Consequently, the ordering of the data changes, so it can no longer be traversed
    /// in the same fashion and becomes non-contiguous.
    /// For instance, consider the following matrix of shape `[2, 3]`, stored as `0 1 2 3 4 5` in memory.
    /// ```ignore
    /// 0 1 2
    /// 3 4 5
    /// ```
    /// Reshaping it to `[3, 2]` yields the matrix below. Notice that, despite its altered shape,
    /// the row-major view of this matrix remains `0 1 2 3 4 5`.
    /// ```ignore
    /// 0 1
    /// 2 3
    /// 4 5
    /// ```
    /// Permuting it, however, results in the following non-contiguous matrix. Although, like the matrix above,
    /// this one is of shape `[3, 2]`, its row-major representation has changed to `0 3 1 4 2 5`,
    /// which differs from how it's actually stored in memory, i.e., `0 1 2 3 4 5`.
    /// ```ignore
    /// 0 3
    /// 1 4
    /// 2 5
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the length of the given ordering doesn't match the tensor's rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![4, 6, 10], 0).permute(&[1, 2, 0]);
    ///
    /// assert_eq!([6, 10, 4], tensor.shape());
    /// assert_eq!([10, 1, 60], tensor.strides());
    /// ```
    pub fn permute(self, ord: &[usize]) -> Self {
        assert_eq!(
            ord.len(),
            self.shape.len(),
            "Order length doesn't equal number of dimensions."
        );

        let mut new_shape = vec![0; self.shape.len()];
        let mut new_strides = vec![0; self.strides.len()];

        for (new_idx, &old_idx) in ord.iter().enumerate() {
            new_shape[new_idx] = self.shape[old_idx];
            new_strides[new_idx] = self.strides[old_idx];
        }

        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Drops a singleton axis in the tensor.
    ///
    /// This function is merely a convenience wrapper around reshape.
    ///
    /// # Panics
    ///
    /// This function panics if the given axis is not a singleton axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1, 2, 3], vec![1, 3]).squeeze(0);
    ///
    /// assert_eq!([3], tensor.shape());
    /// ```
    pub fn squeeze(self, axis: usize) -> Self {
        assert_eq!(self.shape[axis], 1, "Axis isn't a singleton.");

        let mut shape = self.shape.clone();
        shape.remove(axis);
        self.reshape(shape)
    }

    /// Adds a singleton axis to the tensor.
    ///
    /// This function is merely a convenience wrapper around reshape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1, 2, 3], vec![3]).unsqueeze(0);
    ///
    /// assert_eq!([1, 3], tensor.shape());
    /// ```
    pub fn unsqueeze(self, axis: usize) -> Self {
        let mut shape = self.shape.clone();
        shape.insert(axis, 1);
        self.reshape(shape)
    }

    /// Broadcasts the tensor.
    ///
    /// A broadcasted tensor has its elements repeated across singleton dimensions
    /// to match a given shape. Missing axes are implicitly understood to be singletons.
    /// By pretending the strides of singleton axes are 0, this function mimicks this
    /// behaviour without actually creating a larger tensor with repeated elements.
    ///
    /// # Panics
    ///
    /// This function panics if the tensor can't be broadcasted to the given shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::randn(vec![6, 1], 0).broadcast(vec![4, 6, 10]);
    ///
    /// assert_eq!([4, 6, 10], tensor.shape());
    /// ```
    pub fn broadcast(self, shape: Vec<usize>) -> Self {
        Self {
            data: self.data,
            strides: Self::broadcast_strides(&self.strides, &self.shape, &shape),
            shape,
        }
    }

    /// Concatenates the tensor and the other operand along an axis.
    ///
    /// Assuming two tensors have the same dimensions everywhere except for the given axis,
    /// they are joined along that axis. Several applications of this function can be used to
    /// concatenate multiple tensors.
    ///
    /// # Panics
    ///
    /// This function panics if the tensors don't have matching shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![1, 2, 3], vec![1, 3]);
    /// let tensor2 = Tensor::new(vec![4, 5, 6], vec![1, 3]);
    /// let concated = tensor1.concat(&tensor2, 0);
    ///
    /// assert_eq!(Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]), concated);
    /// ```
    pub fn concat(&self, other: &Self, axis: usize) -> Self {
        assert_eq!(
            self.shape.len(),
            other.shape.len(),
            "Tensors aren't of the same rank."
        );

        let mut shape_concated = Vec::with_capacity(self.shape.len());
        for (i, (&dim1, &dim2)) in self.shape.iter().zip(other.shape.iter()).enumerate() {
            if i != axis && dim1 != dim2 {
                panic!("Tensors have different dimensions.");
            }
            shape_concated.push(dim1 + if i == axis { dim2 } else { 0 });
        }

        let n_concated = shape_concated.iter().product();
        let mut concated = Vec::with_capacity(n_concated);
        let strides_concated = Self::comp_strides(&shape_concated);

        for idx_concated in 0..n_concated {
            let mut coords = Self::idx_to_coords(idx_concated, &strides_concated);
            if coords[axis] < self.shape[axis] {
                let idx_self = Self::coords_to_idx(&coords, &self.strides);
                concated.push(self.data[idx_self]);
            } else {
                coords[axis] -= self.shape[axis];
                let idx_other = Self::coords_to_idx(&coords, &other.strides);
                concated.push(other.data[idx_other]);
            }
        }

        Self {
            data: concated,
            shape: shape_concated,
            strides: strides_concated,
        }
    }

    /// Slices the tensor given start (inclusive) and end (exclusive) indices.
    ///
    /// For start indices `[s_1, s_2, ..., s_n]` and end indices `[e_1, e_2, ..., e_n]`,
    /// `n` being the tensor's rank, slicing returns a tensor of shape
    /// `[e_1 - s_1, e_2 - s_2, ..., e_n - s_n]`, where index `[i_1, i_2, ..., i_n]`
    ///  of the sliced tensor correspodns to index `[s_1 + i_1, s_2 + i_2, ..., s_n + i_n]`
    /// from the original tensor. In other words, the sliced tensor comprises every element
    /// whose coordinates fall within start and end.
    ///
    /// # Panics
    ///
    /// This function panics if the tensor's shape, the start indices, and the end indices don't have matching lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::new((1..=18).collect(), vec![2, 3, 3]);
    /// let start = vec![0, 1, 1];
    /// let end = vec![2, 3, 3];
    /// let sliced = tensor.slice(&start, &end);
    ///
    /// assert_eq!(Tensor::new(vec![5, 6, 8, 9, 14, 15, 17, 18], vec![2, 2, 2]), sliced);
    /// ```
    pub fn slice(&self, start: &Vec<usize>, end: &Vec<usize>) -> Self {
        assert_eq!(
            start.len(),
            end.len(),
            "Start and end indices don't have matching lengths."
        );
        assert_eq!(
            self.shape.len(),
            start.len(),
            "Slicing indices don't match the tensor's shape."
        );

        let shape_sliced: Vec<usize> = start.iter().zip(end).map(|(a, b)| b - a).collect();
        let mut data_sliced = Vec::with_capacity(shape_sliced.iter().product());

        let mut stack = vec![(vec![], 0)];
        while let Some((coords, axis)) = stack.pop() {
            if axis == start.len() {
                let idx = Self::coords_to_idx(&coords, &self.strides);
                data_sliced.push(self.data[idx]);
                continue;
            }

            for value in (start[axis]..end[axis]).rev() {
                let mut next = coords.clone();
                next.push(value);
                stack.push((next, axis + 1));
            }
        }

        Self::new(data_sliced, shape_sliced)
    }

    /// Applies an arbitrary unary operation on the tensor.
    ///
    /// The unary operation should accept a scalar and return another.
    /// This function applies it element wise to every value in the tensor,
    /// producing a contiguous tensor of the same shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1, 2, 3], vec![3]);
    /// let res = tensor.apply_uop(|a| 2 * a);
    ///
    /// assert_eq!(Tensor::new(vec![2, 4, 6], vec![3]), res);
    /// ```
    pub fn apply_uop<U>(&self, op: impl Fn(T) -> U) -> Tensor<U>
    where
        U: Copy,
    {
        let n = self.shape.iter().product();
        let mut res = Vec::with_capacity(n);
        let strides_res = Self::comp_strides(&self.shape);

        for idx_res in 0..n {
            let idx_orig = Self::translate_idx(idx_res, &strides_res, &self.strides);
            res.push(op(self.data[idx_orig]));
        }

        Tensor::<U>::new(res, self.shape.clone())
    }

    /// Applies an arbitrary binary operation on the tensor and the other operand.
    ///
    /// When the dimensions of the tensor and the other operand don't match,
    /// they're broadcasted to a common shape. This broadcasting is copy-free
    /// and performed by setting the strides of broadcasted axes to 0.
    /// The binary operation should accept two scalars and return another.
    /// This function applies it to pairs of corresponding elements (i.e., same coordinates)
    /// from the two operands, producing a contiguous tensor with the
    ///
    /// # Panics
    ///
    /// This function panics if the tensors can't be broadcasted to a common shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![5, 10, 15], vec![3]);
    /// let tensor2 = Tensor::new(vec![2, 3, 4], vec![3]);
    /// let res = tensor1.apply_bop(&tensor2, |a, b| a % b);
    ///
    /// assert_eq!(Tensor::new(vec![1, 1, 3], vec![3]), res);
    /// ```
    pub fn apply_bop<U>(&self, other: &Self, op: impl Fn(T, T) -> U) -> Tensor<U>
    where
        U: Copy,
    {
        let shape_broadcasted = Self::broadcast_shapes(&self.shape, &other.shape);
        let strides_self_broadcasted =
            Self::broadcast_strides(&self.strides, &self.shape, &shape_broadcasted);
        let strides_other_broadcasted =
            Self::broadcast_strides(&other.strides, &other.shape, &shape_broadcasted);

        let n_broadcasted = shape_broadcasted.iter().product();
        let mut res = Vec::with_capacity(n_broadcasted);
        let strides_res = Self::comp_strides(&shape_broadcasted);

        for idx_res in 0..n_broadcasted {
            let idx_self = Self::translate_idx(idx_res, &strides_res, &strides_self_broadcasted);
            let idx_other =
                Self::translate_idx(idx_res, &strides_res, &strides_other_broadcasted);
            res.push(op(self.data[idx_self], other.data[idx_other]));
        }

        Tensor::<U> {
            data: res,
            shape: shape_broadcasted,
            strides: strides_res
        }
    }

    /// Applies an arbitrary reduction operation on the tensor at the given axis.
    ///
    /// The reduction operation should accept vectors and reduce them to scalar.
    /// This function applies it to vectors at the given axis and fills in a new
    /// tensor with the results. Concretely, for reduction axis `i`, the output at location
    /// `[a_1, a_2, ..., a_{i-1}, a_{i+1}, ..., a_n]` can be described as
    /// `op` applied over indices `[a_1, a_2, ..., a_i, ..., a_n]` of the input,
    /// where `a_i` indexes the relevant axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// let res = tensor.apply_red(1, |vec| vec[0]);
    ///
    /// assert_eq!(Tensor::new(vec![1, 4], vec![2]), res);
    /// ```
    pub fn apply_red<U>(&self, axis: usize, op: impl Fn(Vec<T>) -> U) -> Tensor<U>
    where
        U: Copy,
    {
        let n_red = self.shape.iter().product::<usize>() / self.shape[axis];
        let mut res = Vec::with_capacity(n_red);

        let mut shape_red = self.shape.clone();
        let shape_axis = shape_red.remove(axis);

        if shape_red.len() == 0 {
            shape_red = vec![1];
        }

        let strides_red = Self::comp_strides(&shape_red);

        let mut strides_orig = self.strides.clone();
        let stride_axis = strides_orig.remove(axis);

        for idx_red in 0..n_red {
            let mut vec = Vec::with_capacity(shape_axis);
            let mut idx_orig = Self::translate_idx(idx_red, &strides_red, &strides_orig);
            for _ in 0..shape_axis {
                vec.push(self.data[idx_orig]);
                idx_orig += stride_axis;
            }
            res.push(op(vec));
        }

        Tensor::<U> {
            data: res,
            shape: shape_red,
            strides: strides_red,
        }
    }
}

/// Implements operations applicable to numeric data in the most general sense.
impl<T> Tensor<T>
where
    T: Add<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + Sub<Output = T>
        + PartialOrd
        + Default
        + Copy,
{
    /// Calculates the vector dot product.
    ///
    /// NumPy's dot product somewhat confusingly accepts inputs of arbitrary rank,
    /// resulting in different operations depending on the shape of the input.
    /// However, this function only covers actual vector dot products, but more advanced
    /// uses can be simulated using a combination of matrix multiplication, element wise products,
    /// and sums.
    ///
    /// # Panics
    /// This function panics of the inputs aren't of rank 1 or don't have matching dimensions.
    ///
    /// # Example
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![1, 2, 3], vec![3]);
    /// let tensor2 = Tensor::new(vec![2, 3, 4], vec![3]);
    /// let res = tensor1.dot(&tensor2);
    ///
    /// assert_eq!(20, res);
    /// ```
    pub fn dot(&self, other: &Self) -> T {
        assert!(
            self.shape.len() == 1 && other.shape.len() == 1,
            "Tensors aren't of rank 1."
        );
        assert_eq!(
            self.shape[0], other.shape[0],
            "Tensors don't have matching dimensions."
        );

        let res = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::default(), |acc, val| acc + val);
        res
    }

    /// Performs matrix multiplication.
    ///
    /// Most array libraries have a more general notion of matrix multiplication
    /// that treats higher-rank tensors as stacks of matrices and broadcasts them accordingly.
    /// However, this function only covers actual matrix multiplication, but more advanced
    /// uses can be simulated using a combination of reshaping and broadcasting.
    ///
    /// # Panics
    /// This function panics of the inputs aren't of rank 2 or don't have matching inner dimensions.
    ///
    /// # Example
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
    /// let res = tensor1.matmul(&tensor2);
    ///
    /// assert_eq!(Tensor::new(vec![19, 22, 43, 50], vec![2, 2]), res);
    /// ```
    pub fn matmul(&self, other: &Self) -> Self {
        assert!(
            self.shape.len() == 2 && other.shape.len() == 2,
            "Tensors aren't of rank 2."
        );
        assert_eq!(
            self.shape[1], other.shape[0],
            "Tensors don't have matching inner dimensions."
        );

        let n_res = self.shape[0] * other.shape[1];
        let mut res = Vec::with_capacity(n_res);
        let shape_res = vec![self.shape[0], other.shape[1]];
        let strides_res = Self::comp_strides(&shape_res);

        for idx in 0..n_res {
            let coords = Self::idx_to_coords(idx, &strides_res);
            let (row, col) = (coords[0], coords[1]);
            let mut acc = T::default();

            for idx_inner in 0..self.shape[1] {
                let val_self = self.data[self.strides[0] * row + self.strides[1] * idx_inner];
                let val_other = other.data[other.strides[0] * idx_inner + other.strides[1] * col];
                acc = acc + val_self * val_other;
            }

            res.push(acc);
        }

        Self {
            data: res,
            shape: shape_res,
            strides: strides_res,
        }
    }

    unary_op!(
        "Takes the absolute value of the tensor element wise.",
        abs,
        |a| if a > T::default() { a } else { -a }
    );
    unary_op!("Negates the tensor element wise.", neg, |a| -a);

    binary_op!("Adds two tensors element wise.", add, |a, b| a + b);
    binary_op!(
        "Divides the tensor by the other operand element wise.",
        div,
        |a, b| a / b
    );
    binary_op!("Multiplies two tensors element wise.", mul, |a, b| a * b);
    binary_op!(
        "Subtracts the tensor by the other operand element wise.",
        sub,
        |a, b| a - b
    );

    binary_op!(
        "Checks if the tensor is greater than the other operand element wise.",
        gt,
        |a, b| a > b,
        bool
    );
    binary_op!(
        "Checks if the tensor is less than the other operand element wise.",
        lt,
        |a, b| a < b,
        bool
    );

    binary_op!(
        "Checks if the tensor is greater than or equal to the other operand element wise.",
        ge,
        |a, b| a >= b,
        bool
    );
    binary_op!(
        "Checks if the tensor is less than or equal to the other operand element wise.",
        le,
        |a, b| a <= b,
        bool
    );

    binary_op!(
        "Takes the maximum of the tensor and the other operand element wise.",
        maximum,
        |a, b| if a > b { a } else { b }
    );
    binary_op!(
        "Takes the minimum of the tensor and the other operand element wise.",
        minimum,
        |a, b| if a < b { a } else { b }
    );

    reduction_op!("Sums the tensor along the given axis.", sum, |vec| vec
        .iter()
        .fold(T::default(), |acc, &val| acc + val));
    reduction_op!(
        "Takes the tensor's product along the given axis.",
        prod,
        |vec| vec.iter().fold(T::default(), |acc, &val| acc * val)
    );

    reduction_op!(
        "Takes the maximum value of the tensor along the given axis.",
        max,
        |vec| vec
            .iter()
            .fold(vec[0], |acc, &val| if acc > val { acc } else { val })
    );
    reduction_op!(
        "Takes the minimum value of the tensor along the given axis.",
        min,
        |vec| vec
            .iter()
            .fold(vec[0], |acc, &val| if acc < val { acc } else { val })
    );
}

/// Implements equality and non-equality for tensors.
impl<T> Tensor<T>
where
    T: PartialEq
        + Copy,
{
    binary_op!(
        "Checks if two tensors are equal element wise.",
        eq,
        |a, b| a == b,
        bool
    );
    binary_op!(
        "Checks if two tensors are unequal element wise.",
        ne,
        |a, b| a != b,
        bool
    );
}

/// Implements operations specific to single-precision tensors.
impl Tensor<f32> {
    /// Generates a normally distributed tensor of the given shape.
    ///
    /// Instead of resorting to external libraries, this function produces random values from scratch.
    /// It repeatedly samples values from the uniform distribution on the unit interval given a seed,
    /// applies the Box-Muller transform to produce normally distributed values,
    /// and updates the seed using a linear congruential generator (LCG) for the subsequent iteration.
    /// The modulus (m), multiplier (a), and increment (c) parameters are respectively
    /// 2^31, 1103515245, and 12345, following glibc.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::randn(vec![6], 0);
    /// let tensor2 = Tensor::randn(vec![2, 3], 1);
    /// ```
    pub fn randn(shape: Vec<usize>, seed: u32) -> Self {
        let n = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut seed = seed;

        for _ in 0..n {
            seed = (1103515245u32.wrapping_mul(seed).wrapping_add(12345)) & 0x7FFFFFFF;
            let u1 = seed as f32 / 0x80000000u32 as f32;
            seed = (1103515245u32.wrapping_mul(seed).wrapping_add(12345)) & 0x7FFFFFFF;
            let u2 = seed as f32 / 0x80000000u32 as f32;

            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            data.push(z);
        }

        Self::new(data, shape)
    }

    unary_op!("Takes the ceiling of the tensor element wise.", ceil, |a| a
        .ceil());
    unary_op!("Takes the floor of the tensor element wise.", floor, |a| a
        .floor());
    unary_op!("Rounds the tensor element wise.", round, |a| a.round());

    unary_op!("Exponentiates the tensor element wise.", exp, |a| a.exp());
    unary_op!(
        "Exponentiates the tensor with base 2 element wise.",
        exp2,
        |a| a.exp2()
    );

    unary_op!(
        "Takes the natural logarithm of the tensor element wise.",
        ln,
        |a| a.ln()
    );
    unary_op!(
        "Takes the common logarithm of the tensor element wise.",
        log10,
        |a| a.log10()
    );
    unary_op!(
        "Takes the binary logarithm of the tensor element wise.",
        log2,
        |a| a.log2()
    );

    unary_op!("Takes the cosine of the tensor element wise.", cos, |a| a
        .cos());
    unary_op!("Takes the sine of the tensor element wise.", sin, |a| a
        .sin());
    unary_op!("Takes the tangent of the tensor element wise.", tan, |a| a
        .tan());

    unary_op!(
        "Takes the hyperbolic cosine of the tensor element wise.",
        cosh,
        |a| a.cosh()
    );
    unary_op!(
        "Takes the hyperbolic sine of the tensor element wise.",
        sinh,
        |a| a.sinh()
    );
    unary_op!(
        "Takes the hyperbolic tangent of the tensor element wise.",
        tanh,
        |a| a.tanh()
    );

    unary_op!(
        "Takes the arccosine of the tensor element wise.",
        acos,
        |a| a.acos()
    );
    unary_op!("Takes the arcsine of the tensor element wise.", asin, |a| a
        .asin());
    unary_op!(
        "Takes the arctangent of the tensor element wise.",
        atan,
        |a| a.atan()
    );

    unary_op!(
        "Takes the square root of the tensor element wise.",
        sqrt,
        |a| a.sqrt()
    );

    binary_op!(
        "Raises the tensor to the power of the other operand element wise.",
        pow,
        |a, b| a.powf(b)
    );

    reduction_op!(
        "Takes the mean of the tensor along the given axis.",
        mean,
        |vec| vec.iter().sum::<f32>() / vec.len() as f32
    );
    reduction_op!(
        "Takes the median of the tensor along the given axis.",
        median,
        |vec| {
            let mut vec = vec;
            vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = vec.len();
            if len % 2 == 0 {
                (vec[len / 2 - 1] + vec[len / 2]) / 2.0
            } else {
                vec[len / 2]
            }
        }
    );
    reduction_op!(
        "Takes the variance of the tensor along the given axis.",
        var,
        |vec| {
            let mean = vec.iter().sum::<f32>() / vec.len() as f32;
            vec.iter().map(|val| (val - mean).powi(2)).sum::<f32>() / vec.len() as f32
        }
    );
}

/// Implements operations specific to boolean tensors.
impl Tensor<bool> {
    /// Returns true if the tensor contains at least one true and false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![false, true, false], vec![3]);
    /// let tensor2 = Tensor::new(vec![false, false, false], vec![3]);
    ///
    /// assert!(tensor1.any());
    /// assert!(!tensor2.any());
    /// ```
    pub fn any(&self) -> bool {
        self.data.iter().any(|&a| a)
    }

    /// Returns true if the tensor contains all trues and false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![true, true, true], vec![3]);
    /// let tensor2 = Tensor::new(vec![true, false, true], vec![3]);
    ///
    /// assert!(tensor1.all());
    /// assert!(!tensor2.all());
    /// ```
    pub fn all(&self) -> bool {
        self.data.iter().all(|&a| a)
    }

    /// Selects items from a true tensor or a false tensor depending on the clause tensor.
    ///
    /// When an element in the clause tensor is true, the value from the true tensor
    /// at the same coordinates is selected. Otherwise, that of the false tensor is chosen.
    ///
    /// # Panics
    ///
    /// This function panics if the shapes of the clause, true, and false tensors don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusten::Tensor;
    ///
    /// let clause = Tensor::new(vec![true, false, true, false], vec![4]);
    /// let if_true = Tensor::new(vec![1, 2, 3, 4], vec![4]);
    /// let if_false = Tensor::new(vec![10, 20, 30, 40], vec![4]);
    /// let res = clause.where_clause(&if_true, &if_false);
    ///
    /// assert_eq!(Tensor::new(vec![1, 20, 3, 40], vec![4]), res);
    /// ```
    pub fn where_clause<T>(&self, if_true: &Tensor<T>, if_false: &Tensor<T>) -> Tensor<T>
    where
        T: Copy,
    {
        assert_eq!(
            if_true.shape, if_false.shape,
            "Shapes of the true and false tensors don't match."
        );
        assert_eq!(
            self.shape, if_true.shape,
            "Shapes of the clause tensor and value tensors don't match."
        );

        let n = self.shape.iter().product();
        let mut res = Vec::with_capacity(n);
        let strides_res = Self::comp_strides(&self.shape);

        for idx_res in 0..n {
            let idx_clause = Self::translate_idx(idx_res, &strides_res, &self.strides);
            let idx_true = Self::translate_idx(idx_res, &strides_res, &if_true.strides);
            let idx_false = Self::translate_idx(idx_res, &strides_res, &if_false.strides);
            res.push(if self.data[idx_clause] {
                if_true.data[idx_true]
            } else {
                if_false.data[idx_false]
            });
        }

        Tensor::new(res, self.shape.clone())
    }
}

arith_trait!("Implements the addition trait for tensors.", Add, add);
arith_trait!("Implements the division trait for tensors.", Div, div);
arith_trait!("Implements the multiplication trait for tensors.", Mul, mul);
arith_trait!("Implements the subtraction trait for tensors.", Sub, sub);

/// Implements the From trait for scalars to tensors.
///
/// Since Rust doesn't support automatic casting, scalars and tensors
/// can't natively interact. Thus, it's necessary to convert scalars
/// into singleton Tensors. This can be cumbersome, and this trait
/// reduces the code burden thanks to `.into()`.
///
/// # Examples
///
/// ```
/// use rusten::Tensor;
///
/// let scalar = 1.into();
///
/// assert_eq!(Tensor::new(vec![1], vec![1]), scalar);
/// ```
impl<T, D> From<D> for Tensor<T>
where
    T: Copy,
    D: Copy + Into<T>,
{
    fn from(value: D) -> Self {
        Tensor::new(vec![value.into()], vec![1])
    }
}

/// Implements equality for tensors.
///
/// # Examples
///
/// ```
/// use rusten::Tensor;
///
/// let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
/// let tensor2 = Tensor::new(vec![1, 3, 2, 4], vec![2, 2]).permute(&[1, 0]);
/// let tensor3 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).permute(&[1, 0]);
///
/// assert_eq!(tensor1, tensor2);
/// assert_ne!(tensor1, tensor3);
/// assert_ne!(tensor2, tensor3);
/// ```
impl<T> PartialEq for Tensor<T>
where
    T: PartialEq
        + Copy,
{
    fn eq(&self, other: &Self) -> bool {
        Tensor::eq(self, other).all()
    }
}
