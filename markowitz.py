import numpy as np
from numpy.linalg import inv

def markowitz(closes, lam = 1):
    '''
    Uses the Markowitz Mean-Variance Portfolio Theory to calculate the optimal weights,
    assuming no risk free assets

    Input:
        closes: matrix (np.array) of closing stock prices for various companies for 5 years
            rows: closing stock prices for a single company
            columns: closing stock prices for a single day
        lam: the value of lambda as mentioned in the paper
            defaults to 1 for testing purposes
    Output:
        optimal weights for each company based on Markowitz Mean-Variance Portfolio Theory

    formulas found here:
    https://sites.math.washington.edu/~burke/crs/408/fin-proj/mark1.pdf
    '''
    # compute returns
    closes = closes.T       
    data_array = (closes[:, 1:] - closes[:, :-1]) / closes[:, :-1]
    means = np.mean(data_array, axis = 1)
    resid = data_array - np.linspace(means, means, data_array.shape[1]).T
    cov = (resid@resid.T)/data_array.shape[1]

    # calculate the inverse (address errors from singular matrices)
    try:
        inv_cov = inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    min_var = np.sum(inv_cov, axis = 1)/np.sum(inv_cov)
    delta = np.sum(inv_cov)*means@inv_cov@means-(np.sum(means@inv_cov))**2
    mew_b = np.dot(min_var, means) + delta*lam/np.sum(inv_cov)
    if np.allclose(means, means[0]*np.ones(means.shape)):
        if (means[0] < mew_b):
            raise Exception("Markowitz is infeasible in this instance.")
        else:
            return min_var
    elif np.dot(min_var, means) >= mew_b:
        return min_var
    else:
        weight_mk = inv_cov@means/np.sum(inv_cov@means)
        v = weight_mk - min_var
        alpha = (mew_b - np.dot(min_var, means))/np.dot(means, v)
        return min_var + alpha*v

