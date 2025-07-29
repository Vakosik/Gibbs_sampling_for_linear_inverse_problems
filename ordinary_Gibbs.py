from utils import *


def gibbs_sampling(M, y, num_samples, omegay_sampling=True, omegax_sampling=True, omegay=1, warm_up=0):
    """
    Samples x from the posterior distribution (p|M,y) if we suppose that p(y|M,x) = N(Mx, (omega_y)^-1) and
    the prior is p(x) = tN(0, (omega_x)^-1; [0,+inf]) where N is the normal distribution.

    Input arguments:
    M                Matrix of size mxn (two dimensions only)
    y                A vector of size m
    num_samples      Number of different x to be sampled
    omegay_sampling  If True, ten omega_y is sampled accorgingly. If False, omega_y is fixedly chosen by another arg.
    omegax_sampling  If True, ten omega_x is sampled accorgingly. If False, omega_x=0.
    omegay           If omegay_sampling is True, then this fixed value is used. If omegay_sampling is False, then
                     it does not have relevance.
    warm_up          Number of first x samples that are omitted due to parameters updates warm up.
                     Therefore, total number of samples is num_samples-warm_up

    output arguments:
    x            Matrix of size (num_samples, n), i.e. each row contains a sample of x
    """
    from scipy.optimize import nnls

    y = y.reshape(y.shape[0], )
    print("shape of M:", M.shape, ", shape of y:", y.shape)

    x = np.zeros((num_samples, M.shape[1]))
    x[0, :] = nnls(M, y)[0]
    print("error of the initial positive least square solution:", np.linalg.norm(np.dot(M, x[0]) - y))

    if omegax_sampling:  # Initial values for sampling omega_x
        shape_0 = 10e-10
        rate_0 = 10e-10
    else:
        omegax = 0

    for i in range(1, num_samples):
        # print("iteration", i)
        if omegay_sampling:
            shape_i = 0.5 * y.shape[0]
            rate_i = 0.5 * np.sum(np.power(np.dot(M, x[i-1]) - y, 2))
            omegay = np.random.gamma(shape_i, 1 / rate_i)

        if omegax_sampling:
            S = 0.5 * np.dot(x[i-1, :], x[i-1, :])
            shape_i = shape_0 + len(x[i-1])/2
            rate_i = rate_0 + S
            omegax = np.random.gamma(shape_i, 1/rate_i)

        for j in range(x.shape[1]):
            if j == 0:
                y_remainder = np.copy(y)
                y_remainder -= np.matmul(M[:, 1:], x[i-1, 1:])
            if j > 0:
                y_remainder += M[:, j] * x[i-1, j] - M[:, j-1] * x[i, j-1]

            if np.dot(M[:, j], M[:, j]) < 1e-20:
                x[i, j] = x[i-1, j]
                continue

            sample_var = 1 / (omegay * np.dot(M[:, j], M[:, j]) + omegax)
            if omegay == 0:
                sample_var = 0

            if sample_var == 0:
                x[i, j] = x[i-1, j]
                continue

            sample_mean = post_dist_mean(y_remainder, M[:, j], omegay, sample_var)
            x[i, j] = truncnorm_ordinary(sample_mean, np.sqrt(sample_var))

    print("sampling terminated")

    x = x[warm_up:]

    return x