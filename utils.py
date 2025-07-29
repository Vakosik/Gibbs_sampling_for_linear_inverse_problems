import numpy as np
from scipy.stats import truncnorm


def post_dist_mean(yr, m, omega_y, sigma2):
    return omega_y * np.dot(yr, m) * sigma2


def bounds_for_alpha(betas, Vj):
    #  if Vj[k] < 0, then we have upper bound conditions. From these, we need to choose the minimum
    #  as the overall upper bound.
    upper_bounds = [betas[k] / np.abs(Vj[k]) for k in range(betas.shape[0]) if Vj[k] < 0]

    if len(upper_bounds) > 0:
        upper_bound = min(upper_bounds)
    else:  # if there is no upped bound condition, we choose a huge number
        upper_bound = 1e20

    #  if Vj[k] > 0, then we have upper bound conditions. From these, we need to choose the minimum
    #  as the overall upper bound.
    lower_bounds = [-betas[k] / Vj[k] for k in range(betas.shape[0]) if Vj[k] > 0]

    if len(lower_bounds) > 0:
        lower_bound = max(lower_bounds)
    else:
        lower_bound = -1e20

    return upper_bound, lower_bound


def truncnorm_ordinary(mean, scale):
    a, b = (0-mean)/scale, np.inf
    s = truncnorm.rvs(a, b, loc=mean, scale=scale, size=1)[0]
    if s < 0:  # for numerical reasons, it might happen that s is not well sampled by truncnorm
        s = 0

    return s


def truncnorm_orthogonal(mean, scale, betas, Vj):
    upper_bound, lower_bound = bounds_for_alpha(betas, Vj)
    a, b = (lower_bound - mean) / scale, (upper_bound - mean) / scale

    # If truncnorm has extremely thin interval to sample in, the function might fail to sample inside the interval.
    # In this case we just select the middle number of the interval
    try:
        s = truncnorm.rvs(a, b, loc=mean, scale=scale, size=1)[0]
    except:
        s = np.mean(np.array([upper_bound, lower_bound]))
    if s > upper_bound or s < lower_bound:
        s = np.mean(np.array([upper_bound, lower_bound]))

    return s