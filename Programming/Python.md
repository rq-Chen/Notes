# Python Notes

## Numpy

- See https://numpy.org/doc/stable/user/numpy-for-matlab-users.html for the difference between numpy and matlab.
- Most important differences:
  - Arrays are passed **by reference**, including all regular **indexing**, which only creates views
    - But `ix_()` generates deep copy
  - Arithmatic operations are **element-wise**, though `@` represents matrix multiplication ('element-wise' is just like that in matlab)
  - **Zero-based, row-first** indexing
    - For example, 2D numpy array is actually a list of rows
    - So `x.flatten()` generates a vector in **row-first** way, while `x.flatten('F')` is in column-first way just as in Matlab
    - Some functions are also different, e.g. `numpy.corrcoef()` treats each row as an variable by default while Matlab always treats each row as an observation
  - When you extract a column from a matrix by indexing, it will become a **1D vector**, and by default viewed as a **row vector**
    - But `np.ix_()` will generate the expected output, i.e. a column vector
- Size:
  - `size()` is the same as `numel()` in matlab, `arr.shape()` is the same as `size(arr)` in matlab
  - Normal one-dimensional Python lists and 1D numpy arrays are **row vectors** by default, but can be implicitly transposed to column vectors (but numpy arrays have fixed shape)
    - e.g., if you multiply an 1D array of size `(n,)` with a column vecor of size `(n, 1)`, the result is an **n * n** matrix rather than a vector
  - Just like Matlab, Numpy will add or remove the trailing singular dimension automatically, but unlike Matlab, it will be removed only after operation rather than immediately after your declaration.
- Indicing:
  - Sequential indicing:
    - Python uses zero-based indicing and the second argument in the `::` command means the position after the end
    - `arr[a-1:b:step, ...]` equals to `arr(a:step:b, :, :)` in matlab (maybe more `, :`) if `step` is positive, and `arr(a:step:b+2, :, :)` if `step` is negative
  - Vector or logical indicing:
    - `arr[np.ix_(vec1, vec2)]` equals to `arr(vec1, vec2)` in matlab
    - **Do not** mix up `ix_()` with regular indicing
- Vectorization:
  - Numpy operations are vectorized in a similar way as Matlab's element-wise operation (that is, arrays must be in "compatible size")
- Others:
  - You can use `+=` in python
  - You must close a file opened by `np.load()` after extracting the data

