from utils import *


def orthogonal_gibbs_sampling(M, y, num_samples, omegay_sampling=True, omegax_sampling=True, omegay=1, warm_up=0):
    """
    Samples x from the posterior distribution (p|M,y) if we suppose that p(y|M,x) = N(Mx, (omega_y)^-1) and
    the prior is p(x) = tN(0, (omega_x)^-1; [0,+inf]) where N is the normal distribution.
    The orthogonal transformation is made by SVD decomposition of M for more efficient sampling.

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
    x            Matrix of size (num_samples, n), i.e. each row contains a sampeled solution of Mx=y
    """
    from scipy.optimize import nnls

    U, S, Vh = np.linalg.svd(M)  # SVD decomposition of M
    V = np.transpose(Vh)

    if M.shape[0] >= M.shape[1]:  # S is a vector, we want to get a matrix Sigma with singular values on the main diag.
        Sigma = np.eye(M.shape[0], M.shape[1]) * S
    else:
        Sigma = np.transpose(np.eye(M.shape[0], M.shape[1])) * S
        Sigma = np.transpose(Sigma)

    A = np.matmul(U, Sigma)
    y = y.reshape(y.shape[0], )

    print("shape of M:", M.shape, ", shape of y:", y.shape)
    print("shape of U:", U.shape, ", shape of Sigma:", Sigma.shape, ", shape of V:", V.shape, ", shape of A:", A.shape)

    x = np.zeros((num_samples, M.shape[1]))
    alphas = np.zeros((num_samples, M.shape[1]))

    x[0, :] = nnls(M, y)[0]
    print("error of the initial positive least square solution:", np.linalg.norm(np.dot(M, x[0]) - y))
    if omegay_sampling:
        print("omega_y will be updated after each full sample of x.")

    # SVD decomposition can produce numerical errors and therefore transformation from x to alpha and back to x does not
    # need to produce non-negative values only. However, we need x non-negative, otherwise, the algorithm may fail.
    # We try to add very small constant to x before transforming to alpha. If it does not help, the exception is raised.
    alphas[0, :] = np.matmul(Vh, x[0, :])
    alpha_to_x = np.matmul(V, alphas[0, :])
    if not np.all(alpha_to_x >= 0):
        alphas[0, :] = np.matmul(Vh, x[0, :]+1e-4)
        alpha_to_x = np.matmul(V, alphas[0, :])
        if not np.all(alpha_to_x >= 0):
            raise Exception("Transformation from x to alpha and back to x does not yield non-negative values due to"
                            "numerical errors."
                            "Check the code and write a solution of your preference.")

    if omegax_sampling:  # Initial values for sampling omega_x
        shape_0 = 10e-10
        rate_0 = 10e-10
    else:
        omegax = 0

    for i in range(1, num_samples):  # loop over samples of x (vectors)
        # print("iteration", i)
        if omegay_sampling:
            shape_i = 0.5 * y.shape[0]
            rate_i = 0.5 * np.sum(np.power(np.dot(M, x[i-1]) - y, 2))
            omegay = np.random.gamma(shape_i, 1 / rate_i)

        if omegax_sampling:
            S = 0.5 * np.dot(alphas[i-1, :], alphas[i-1, :])
            shape_i = shape_0 + len(alphas[i-1])/2
            rate_i = rate_0 + S
            omegax = np.random.gamma(shape_i, 1/rate_i)

        for j in range(x.shape[1]):  # loop over entries of x sample number i
            if j == 0:
                betas = np.matmul(V[:, 1:], alphas[i-1, 1:])
            else:  # computationally economical update, no need to do another matmul for other entries
                betas += V[:, j-1] * alphas[i, j-1] - V[:, j] * alphas[i-1, j]

            # if a_j^T.a_j is very small, then the inversion for sigma might cause troubles
            if np.abs(np.dot(A[:, j], A[:, j])) < 1e-20:
                alphas[i, j] = alphas[i-1, j]
                continue

            sample_var = 1 / (omegay * np.dot(A[:, j], A[:, j]) + omegax)
            if omegay == 0:
                sample_var = 0

            if sample_var == 0:
                alphas[i, j] = alphas[i-1, j]
                continue

            sample_mean = post_dist_mean(np.copy(y), A[:, j], omegay, sample_var)
            alphas[i, j] = truncnorm_scipy(sample_mean, np.sqrt(sample_var), betas, V[:, j])

        x[i, :] = np.matmul(V, alphas[i, :])

    print("sampling terminated")

    x = x[warm_up:]

    return x
