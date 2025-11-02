from data_preprocessing import *

def monte_carlo_price_simulations(prices,log_returns, num_sim_dates, num_price_sims):
    '''
    Runs a given number of price simulations for the given number of dates for
        each ticker
    
    Input:
        log_returns - np.array of the log returns of each asset in the portfolio
                    shape: (num_dates-1) x num_tickers
        num_sim_dates - int for the number of future dates to randomly generate
        num_price_sims - int for the number of random samples to generate
    Output:
        simulated_prices - np.array for the given number of simulations, given number
            of dates, and number of tickers
                    shape: num_sim_dates x num_simulations x num_tickers

    The formulas in my approach are based on the following article:
    https://www.investopedia.com/terms/m/montecarlosimulation.asp
    '''
    # compute prerequisite calculations
    # all have shape num_tickers x 1
    mu = get_mean_returns(log_returns)              # mean returns
    sigma_squared = np.var(log_returns,axis=0)      # variance
    sigma = np.sqrt(sigma_squared)                  # standard deviation
    drift = mu - (sigma_squared/2)                  # drift
    num_dates = log_returns.shape[0]
    num_tickers = log_returns.shape[1]
    shapes = [item.shape for item in [mu,sigma_squared,sigma,drift]]
    assert all(s == (num_tickers,) for s in shapes)

    # simulate future prices for each ticker
    # shape: num_sim_dates x num_price_sims x num_tickers
    last_prices = prices[-1]
    simulated_prices = np.zeros((num_sim_dates,num_price_sims, num_tickers))
    for ticker_idx in range(num_tickers):
        # random_value_i = sigma_i * random number from N(0,1)
        # random_values size: num_sim_dates x num_price_sims
        random_values = sigma[ticker_idx] * np.random.normal(0,1,(num_sim_dates,num_price_sims))
        
        # new_price = prev_price * e^(drift + random_value)
        exp_term = np.exp(drift[ticker_idx] + random_values)
        # combine the exponential terms from prev steps
        # shape: num_sim_dates x num_price_sims
        exp_terms = np.cumprod(exp_term,axis=0)   
        simulated_prices[:,:,ticker_idx] = last_prices[ticker_idx] * exp_terms

    return simulated_prices




# testing
# def main():
#     csv_file = 'stock_details_5_years.csv'

#     close_data, dates, tickers = read_raw_csv(csv_file)
#     dates_subset = dates[:10]
#     tickers_subset = tickers[:10]
#     close_subset = close_data[close_data['Date'].isin(dates_subset)]
#     close_subset = close_data[(close_data['Date'].isin(dates_subset)) & 
#                                 (close_data['Company'].isin(tickers_subset))]
#     prices = get_prices_matrix(close_subset,dates_subset,tickers_subset)

#     log_returns = get_log_returns(prices)
#     mu = get_mean_returns(log_returns)
#     Sigma = get_covariance_matrix(log_returns,mu)

#     simulations = monte_carlo_simulation(prices,log_returns,10,20)

#     print(simulations.shape)

#     return 0


# if __name__ == "__main__":
#     main()