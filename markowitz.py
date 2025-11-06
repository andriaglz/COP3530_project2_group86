import numpy as np
from numpy.linalg import inv

def markowitz(closes):
    # compute returns (you don't need close and open; you just need previous day close)
    data_array = (closes[:, 1:] - closes[:, :-1]) / closes[:, :-1]
    lam = 1
    means = np.mean(data_array, axis = 1)
    resid = data_array - np.linspace(means, means, data_array.shape[1]).T
    cov = (resid@resid.T)/data_array.shape[1]
    inv_cov = inv(cov)
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


# testing
from data_preprocessing import *
def main():
    csv_file = 'stock_details_5_years.csv'
    close_data, dates, tickers = read_raw_csv(csv_file)
    dates_subset = dates[:10]
    tickers_subset = tickers[:10]
    close_subset = close_data[close_data['Date'].isin(dates_subset)]
    close_subset = close_data[(close_data['Date'].isin(dates_subset)) & 
                                (close_data['Company'].isin(tickers_subset))]
    prices = get_prices_matrix(close_subset,dates_subset,tickers_subset)

    try:
        weights = markowitz(prices)
        print("Optimal Weights:\n", weights)
        print("Sum of weights:", np.sum(weights))
    except Exception as e:
        print("Error:", e)
    return 0


if __name__ == "__main__":
    main()