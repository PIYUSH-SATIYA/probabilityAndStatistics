Download face images by running the face image scrapper script from utils/ which scraps from thispersondoesnotexist.com

# Cell 1: loading img data into a data matrix

imread() method reads binary image files and returns a numpy array. But it is still 2D array.
We can convert it to a 1D array (column vector) using flatten() method.

Then store it into x = data matrix

> The dimension of x will be (number of pixels in each image) x (number of images), here (256\*256) x 25 = 65536 x 25

> data matrix X is shaped (65536, 25)
> Each column is a flattened image.(sample)
> Each row is the intensity of one particular pixel across all 25 images. (feature)

# Cell 2: calculating mean and centering the data

since each row is a particular pixel across all images, we calculate the mean of each row (i.e. mean intensity of that pixel across all images) and not of a col. which doesn't make sense.

all the methods are then standard numpy methods, e.g. np.mean() etc.

# Cell 3: calculating covariance matrix

since each of the pixel of the centered data matrix is a random variable and each of the sample is also a random variable, we have two choices to calculate covariance matrix:

1. `C_x = (1 / (n − 1)) * X_centered.T @ X_centered`

    - Shape: (25, 25)

    - Interpretation: Covariance between samples

This is what you're aiming for — low-dimensional, fast, and perfect for computing eigenfaces when n < m (which is your case)

2. `C_x = (1 / (n − 1)) * X_centered @ X_centered.T`

    - Shape: (65536, 65536) 😵

    - Interpretation: Covariance between pixels

This is mathematically correct but impossible to compute with limited RAM

when we compute the eigen vectors of 25x25 matrix it can be later projected on the original space to get the eigenfaces (aka eigen vectors or principal components). This is a trick or called dual PCA.

so we are going with option 1. and for it

we are actually calculating:

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

# cell next: projceting and reconstructing the data with choosen k.

1. Why do we multiply X_centered @ W_small to get eigenfaces?

-   W_small is an eigenvector of the dual covariance matrix C_x = X_centered.T @ X_centered / (n−1) of size (25×25).

-   But we want eigenvectors of the original covariance X_centered @ X_centered.T / (n−1), which is (65536×65536).

-   There's a theoretical result:

    -   If v is an eigenvector of the small (n×n) covariance matrix with eigenvalue λ, then u = X_centered @ v is an eigenvector of the original large covariance matrix.

-   So X_centered @ W_small gives us the eigenfaces (columns of U) in the original 65536-dimensional pixel space.

2. Why do we then do U.T @ X_centered to get the projected data?

-   U has shape (65536, k) — each column is an eigenface.

-   X_centered has shape (65536, 25) — each column is a centered image.

-   To get the coordinates of each image in the reduced k-dimensional space, you project the centered images onto these eigenfaces:

-   X_projected = U.T @ X_centered

-   Resulting shape: (k, 25) → each column is a k-dimensional representation of the original image.

These are what we will use to reconstruct the images later.

---

✅ Quick intuition:

-   X_centered @ W_small → "build eigenfaces in pixel space"

-   U.T @ X_centered → "express each original image in terms of eigenfaces"
