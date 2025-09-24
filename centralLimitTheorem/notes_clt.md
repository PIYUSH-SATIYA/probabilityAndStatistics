# Central Limit Theorem

The Central Limit Theorem states that the sum (or average) of a large number of independent random variables approaches a normal distribution, regardless of the original distributions' shapes.

# project specifications

-   taking 10 samples of size 100
    -   first identically distributed (exponential)
    -   second non-identically distributed (exponential and uniform)

## 1 - Identically Distributed Samples

in this one, we have independent and identically distributed samples, (here all exponential).

we are generating several such samples (say n_distributions), each of size n_points.

we sum all those random variables and get a new one, standardize it and calculate its mean and standard deviation.
As expected, it approaches a normal distribution.

Two important observations:

1. As we increase no. of distributions, the mean of the standardized sum approaches 0 and standard deviation approaches 1, however the bell curve may not be very smooth for small no. of distributions.
2. The bell curve becomes smoother as we increase the no. of points in each distribution.

The mean and standard deviation of the standardized data is also not an measure of how close the distribution is to a normal distribution. For that, we can use statistical tests like the Kolmogorov-Smirnov test or the Anderson-Darling test, or simply the shape of the bell curve.

same goes for mean and standard deviation of the summed data.
