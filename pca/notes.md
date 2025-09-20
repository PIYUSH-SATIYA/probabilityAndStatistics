Download face images by running the face image scrapper script from utils/ which scraps from thispersondoesnotexist.com

# Cell 1: loading img data into a data matrix

`imread()` method reads binary image files and returns a numpy array. But it is still 2D array.
We can convert it to a 1D array (column vector) using `flatten()` method.

Then store it into x = data matrix

> The dimension of x will be (number of pixels in each image) x (number of images), here (256\*256) x 25 = 65536 x 25

> data matrix X is shaped (65536, 25)
> Each column is a flattened image.(sample)
> Each row is the intensity of one particular pixel across all 25 images. (feature)

# Cell 2: calculating mean and centering the data

Since each row is a particular pixel across all images, we calculate the mean of each row (i.e. mean intensity of that pixel across all images) and not of a col. which doesn't make sense.

All the methods are then standard numpy methods, e.g. `np.mean()` etc.

# Cell 3: calculating covariance matrix

Since each of the pixel of the centered data matrix is a random variable and each of the sample is also a random variable, we have two choices to calculate covariance matrix:

1. `C_x = (1 / (n − 1)) * X_centered.T @ X_centered`

    @ meaning matrix multiplication, and since we are working with the centered data matrix X_centered we don't have to subtract the mean again. also the inner product takes care of the summation.

    - Shape: (25, 25)
    - Interpretation: Covariance between samples

    This is what you're aiming for — low-dimensional, fast, and perfect for computing eigenfaces when n < m (which is your case)

2. `C_x = (1 / (n − 1)) * X_centered @ X_centered.T`

    - Shape: (65536, 65536) 😵
    - Interpretation: Covariance between pixels

    This is mathematically correct but impossible to compute with limited RAM

When we compute the eigen vectors of 25x25 matrix it can be later projected on the original space to get the eigenfaces (aka eigen vectors or principal components). This is a trick or called dual PCA.

So we are going with option 1. and for it

We are actually calculating:

```python
C[i][j]=(1/(n−1))∑​Xcentered​[k][i]⋅Xcentered​[k][j]
(summation over k from 1 to n)

that is
n = 3  # number of samples
d = 2  # number of features

C = np.zeros((n, n))
for i in range(n):
     for j in range(n):
          dot = 0
          for k in range(d):
                dot += X_centered[k][i] * X_centered[k][j]
          C[i][j] = dot / (d - 1)
```

# Cell 4 and next: projecting and reconstructing the data with chosen k

## Why X_centered @ W_small gives eigenfaces in original pixel space

In dual PCA, we compute the smaller covariance matrix:

$$C_{dual} = \frac{1}{n-1} X_{centered}^T X_{centered} \text{ (shape: } (n \times n)\text{)}$$

where $n = 25$ images, and each image is flattened into a vector of size $d = 65536$.

We get eigenvectors $v_i \in \mathbb{R}^n$ of this dual covariance.

But we want eigenvectors in the original pixel space, i.e., vectors of shape $d = 65536$, to represent "eigenfaces".

**📌 There's a theoretical result:**

If $v$ is an eigenvector of the dual covariance matrix with eigenvalue $\lambda$, then

$$u = X_{centered} v$$

is an eigenvector of the original covariance matrix

$$C = \frac{1}{n-1} X_{centered} X_{centered}^T$$

with the same eigenvalue $\lambda$, and shape $d \times 1$.

Hence:

```python
u = X_{centered} @ W_{small}  # shape: (65536, k)
```

gives us eigenfaces — one per column — in the original pixel space.

## Why X_centered @ W_small gives the projection (X_projected)

Each eigenface (column of U) is a direction in the pixel space.
Projecting each centered image onto these eigenfaces gives coordinates in R^k.

This is done via:

```python
X_projected = X_centered @ W_small  # shape: (65536, k)
```

Note: This works under dual PCA because we're treating each image as a random variable.
The matrix multiplication acts as both eigenface computation and projection step.  
For each pixel as a random variable, this gives the projection of each image onto the k eigenfaces and then we can reconstruct the images. as the notes commented out below

where:

-   X_centered: shape (65536, 25)
-   W_small: shape (25, k)

Output: each row is the projection of an image in reduced k-dimensional space.

## ✅ Why X_projected @ W_small.T + mean reconstructs the images

To go back to pixel space, we do the reverse operation:

X_reconstructed = X_projected @ W_small^T + mean

In code:

```python
X_reconstructed = X_projected @ W_small.T + mean
```

-   X_projected: shape (65536, k)
-   W_small^T: shape (k, 25)

Output: shape (65536, 25), i.e., each image vector is reconstructed in pixel space.

We then reshape each row of X_reconstructed into 64×64 to visualize the reconstructed image.

## ✅ Quick Intuition Summary

| Operation                      | What it does                      | Why it works                                             |
| ------------------------------ | --------------------------------- | -------------------------------------------------------- |
| X_centered @ W_small           | Compute eigenfaces in pixel space | Projects small eigenvectors to high-dimensional space    |
| X_projected @ W_small.T + mean | Reconstruct images                | Combines k eigenfaces back to approximate original image |

<!-- Notes by claude not sure about correctness, it is for pixel as a random variable -->

<!-- Download face images by running the face image scrapper script from utils/ which scraps from thispersondoesnotexist.com

# Corrected PCA Notes for Face Recognition

## Cell 1: Loading Image Data into Data Matrix

`imread()` method reads binary image files and returns a numpy array. But it is still 2D array.
We can convert it to a 1D array (column vector) using `flatten()` method.

Then store it into x = data matrix

> The dimension of x will be (number of pixels in each image) x (number of images), here (256×256) x 25 = 65536 x 25

> **Data matrix X is shaped (65536, 25)**
>
> -   **Each column is a flattened image (sample)**
> -   **Each row is the intensity of one particular pixel across all 25 images (feature)**

## Cell 2: Calculating Mean and Centering the Data

Since each **column** represents an image sample, we calculate the mean **across columns (axis=1)** to get the average pixel intensity for each pixel position across all images.

```python
mean = np.mean(x, axis=1, keepdims=True)  # shape: (65536, 1)
```

This gives us the "average face" - the mean intensity value for each of the 65536 pixel positions.

## Cell 3: Calculating Covariance Matrix (Dual PCA Approach)

In standard PCA, we would compute the covariance between **features (pixels)**:

$$C = \frac{1}{n-1} X_{centered} X_{centered}^T$$

-   Shape: (65536, 65536) 😵
-   This is computationally intractable for large images

**Dual PCA Trick:**
When the number of samples (n=25) is much smaller than the number of features (d=65536), we use:

$$C_{dual} = \frac{1}{n-1} X_{centered}^T X_{centered}$$

-   Shape: (25, 25) ✅
-   This computes covariance between **samples**, not pixels
-   Much more efficient to compute eigendecomposition

```python
c_x = (1 / (n - 1)) * (x_centered.T @ x_centered)  # shape: (25, 25)
```

## Cell 4: Eigendecomposition and Dual-to-Original Space Conversion

We compute eigenvalues and eigenvectors of the dual covariance matrix:

```python
eigenvalues, eigenvectors = np.linalg.eig(c_x)  # eigenvectors shape: (25, k)
```

**Key Theoretical Result:**
If $v$ is an eigenvector of the dual covariance $C_{dual}$ with eigenvalue $\lambda$, then:

$$u = X_{centered} v$$

is an eigenvector of the original covariance matrix $C$ with the same eigenvalue $\lambda$.

## Cell 5: Projection and Reconstruction

### ❌ Error in Your Code and Understanding

Your code has this line:

```python
x_projected = x_centered @ w_small  # WRONG!
```

This is **incorrect**. Here's what's actually happening:

### ✅ Correct Understanding

```python
# Step 1: Convert dual eigenvectors to original space (eigenfaces)
eigenfaces = x_centered @ w_small  # shape: (65536, k)
# These are the actual eigenfaces in pixel space

# Step 2: Project original data onto eigenfaces
x_projected = x_centered.T @ eigenfaces  # shape: (25, k)
# OR equivalently: x_projected = (x_centered.T @ x_centered) @ w_small
```

However, your reconstruction code accidentally works because:

```python
x_reconstructed = x_projected @ w_small.T + mean
```

When you do `(x_centered @ w_small) @ w_small.T`, this becomes:
`x_centered @ (w_small @ w_small.T)`

Since `w_small` contains orthonormal eigenvectors, `w_small @ w_small.T` approximates the identity matrix (for the top k components), so this approximately reconstructs the centered data.

### ✅ What Each Step Actually Does

| Variable                    | Shape       | What it represents                                                |
| --------------------------- | ----------- | ----------------------------------------------------------------- |
| `x_centered @ w_small`      | (65536, k)  | **Eigenfaces** - principal directions in pixel space              |
| `x_centered.T @ eigenfaces` | (25, k)     | **Projected data** - coordinates of each image in eigenface space |
| `projected @ eigenfaces.T`  | (25, 65536) | **Reconstructed** - images reconstructed from k components        |

### ✅ Corrected Interpretation

1. **Eigenfaces**: `x_centered @ w_small` gives you the actual eigenfaces (principal components in pixel space)
2. **Projection**: To project data, you compute coordinates in the eigenface basis
3. **Reconstruction**: Combine eigenfaces with their coefficients to reconstruct images

## Summary of Key Corrections

1. **Dual PCA** computes covariance between samples, not pixels
2. Your `x_projected` variable actually contains **eigenfaces**, not projections
3. The reconstruction works due to the orthogonality properties of eigenvectors
4. The true projection would be `x_centered.T @ eigenfaces`

Your code produces correct results, but the variable naming and conceptual understanding need correction! -->
