# Python Notes

## Anaconda

How to install python 2.7 (32 bit) and python 3.7 (64 bit) simultaneously on a 64 bit windows 10 machine:

1. Download and install [Anaconda 3](https://www.anaconda.com/distribution/) (using default settings).

   Anaconda helps maintain Python packages and enables switching between different python version through virtual environments. In my case, I create two environment other than the normal one (base): py27win32 for NAO Robot development and tensorflow2gpu for CNN training (python 3.7 (64 bit), tensorflow2, keras).

2. Add Anaconda directories to the system path:

   1. `C:\Users\YOURNAME\Anaconda3`
   2. `C:\Users\YOURNAME\Anaconda3\Scripts`
   3. `C:\Users\YOURNAME\Anaconda3\Library\bin`

3. Create a virtual environment for 32-bit python 2.7:

   Enter the command in cmd or Anaconda prompt (replace YOURENVNAME with the name your want, e.g. `py27Win32`):

   ```shell
   conda create -n YOURENVNAME
   conda activate YOURENVNAME
   conda config --env --set subdir win-32
   conda install python=2.7
   conda deactivate
   ```

   The `subdir` key enables us to force Anaconda to search only for win32 packages.

4. Register the new python:

   Enter `regedit`, go to `HKEY_CURRENT_USER\Software\Python\PythonCore`, create a new folder `2.7` in the style of the original one (`3.7`), changing all directory prefix to `ANACONDAPTH\envs\YOURENVNAME`.



## Jupyter Notebook

- Magic code for inline illustration: `%matplotlib inline`
  - Use matplotlib instead of opencv for illustration, since the latter will pump up new window
- Run each block to see the output (including markdown blocks)
- If you want to change the path of the Notebook, the easiest way is to cd into the path in a shell, then type `jupyter notebook` to open it. (And **don't** close the shell)
- If you want to use vscode to edit Jupyter file, make sure you launch vscode by **Anaconda Shell** (type `code`) rather than Navigator! (And **don't** close the shell)
  - For code completion:
    - Install *IntelliCode* and *Pylance* from vscode marketplace
    - Open a .py file, then a dialogue will pump up asking you whether you want to set *Pylance* as the default language server, click 'yes and reload'
    - Open an iPython file and *Intellicode* will begin downloading model, and work immediately



## Python

- For Python 2 only: `/` is the same as the `/` in C, different from that in matlab or python 3.



## Numpy

- See https://numpy.org/doc/stable/user/numpy-for-matlab-users.html for the difference between numpy and matlab.
- Most important differences:
  - Arrays are passed **by reference**, including all regular **indexing**, which only creates views
    - Use `Y = X.copy()` for deep copy
    - `ix_()` generates deep copy
  - Arithmatic operations are **element-wise**, though `@` represents matrix multiplication ('element-wise' is just like that in matlab)
    - But `dot()` conducts matrix multiplication if both inputs are matrices!
  - `np.std()` and `np.var()` are normalized by **n** rather than **n - 1** in Matlab
  - **Zero-based, row-first** indexing
    - For example, 2D numpy array is actually a list of rows
    - So `x.flatten()` generates a vector in **row-first** way, while `x.flatten('F')` is in column-first way just as in Matlab
    - Some functions are also different, e.g. `numpy.corrcoef()` treats each row as an variable by default while Matlab always treats each row as an observation
  - When you extract a column from a matrix by indexing, it will become a **1D vector**, and by default viewed as a **row vector**
    - But `np.ix_()` will generate the expected output, i.e. a column vector
  - A silly difference:
    - `np.round()` (or `np.floor()`, `np.ceil()`) returns a **numpy.float64** object (more exactly, the same type as the input) which cannot be used directly as an index
    - More silly difference: `round(np.array(0))` also returns a **numpy.float64** object!
    - The workaround is to use `array.astype('int')`, which converts each element to int (and preserves the shape)
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
  - You must close a file opened by `np.load('xxx.npz')` after extracting the data (but for `.npy` file the return value is a single array)
  - Somtimes `np.load('xxx.npy')` may return something like `tmp1 = array({'a', va, 'b', vb, ...}, dtype=object)`, which is an 0-dimensional array containing a dict. In order to get the contain, you have to use `tmp = tmp1[()]` or `tmp = tmp1.item()`
  - There's no `find()` in numpy, but you can use `nonzero()`. 
    - **CAUTION**: `np.nonzero()` returns a tuple of arrays, each containing the indices of none-zero elements in corresponding dimension!



## Matplotlib

- `plt.clim(a, b)`, not `clim([a, b])`
- `plt.subplot()` cannot combine several small subplots like matlab
  - But you can use `plt.subplot2grid(size, loc, rowspan, colspan)`



## OpenCV

- The top-left pixel is (0,0) and the y axis points to the bottom.
- OpenCV's default color space is **BGR!!!** And the range will vary too (0 to 255 for CV_8U images, 0 to 65536 for CV_16U images, 0-1 for CV_32F images). See https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html.
- `cv2.imshow()` will only show image after execution of `cv2.waitKey()`, similarly `plt.imshow()` or other matplotlib functions will only show image after `plt.show()`.
- `cv2.HOUGH_GRADIENT` is called `cv2.cv.CV_HOUGH_GRADIENT` in the old version.

