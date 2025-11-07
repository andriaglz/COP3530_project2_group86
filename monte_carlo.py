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
    num_tickers = log_returns.shape[1]
    shapes = [item.shape for item in [mu,sigma_squared,sigma,drift]]
    assert all(s == (num_tickers,) for s in shapes)

    # simulate future prices for each ticker
    # shape: num_sim_dates x num_price_sims x num_tickers
    last_prices = prices[-1]
    simulated_prices = np.zeros((num_sim_dates,num_price_sims, num_tickers))
    random_values = sigma * np.random.normal(0,1,(num_sim_dates,num_price_sims,num_tickers))
    exp_term = np.exp(drift + random_values)
    exp_terms = np.cumprod(exp_term,axis=0) 
    simulated_prices = last_prices * exp_terms  
    return simulated_prices

def monte_carlo_weights(simulated_prices, num_weight_simulations):
    # simulated prices shape: num_sim_dates x num_price_sims x num_tickers
    assert len(simulated_prices.shape) == 3
    _, num_price_sims, num_tickers = simulated_prices.shape

    # randomly sample weights from a standard normal distribution
    # weights in the portfolio should sum to 1
    # weights shape: (num_weight_simulations, num_tickers)
    weights = np.random.normal(0,1,(num_weight_simulations,num_tickers))
    weights = weights / np.sum(weights,axis=1,keepdims=True)

    # duplicate the weights to be consistent across all the dates for each price simulation
    # final weights shape: num_weight_simulations x num_price_sims x num_tickers
    final_weights = np.expand_dims(weights,axis=1)
    final_weights = np.broadcast_to(final_weights,(num_weight_simulations,num_price_sims,num_tickers))

    assert np.allclose(np.sum(final_weights,axis=2),1)

    return final_weights

def monte_carlo_optimal_weights(price_simulations,weight_simulations,rfr=0.03):
    # weights shape: num_weight_simulations x num_price_sims x num_tickers
    num_weight_sims,num_price_sims,num_tickers = weight_simulations.shape
    # prices shape: num_sim_dates x num_price_sims x num_tickers
    num_sim_dates = price_simulations.shape[0]
    
    # convert simulated prices to log returns to allow multiplication
    # log returns shape: num_sim_dates-1 x num_price_sims x num_tickers
    log_price_returns = np.log(price_simulations[1:,:,:]/price_simulations[:-1,:,:])
    
    # multiply the weights by the returns to calculate portfolio returns
    weights = weight_simulations.reshape(num_weight_sims,1,num_price_sims,num_tickers)
    returns = log_price_returns.reshape(1,num_sim_dates-1,num_price_sims,num_tickers)
    # weights must be multiplied across all returns across all dates
    # individual returns shape : num_weight_sims x num_sim_dates x num_price_sims x num_tickers
    individual_returns = weights * returns
    # portfolio returns shape: num_weight_sims x num_sim_dates-1 x num_price_sims
    portfolio_returns = np.sum(individual_returns,axis=-1)
    assert portfolio_returns.shape == (num_weight_sims,num_sim_dates-1,num_price_sims)
    
    # calculate the sharpe for each portfolio
    # sharpe = (mu-rfr)/sigma = (average_return - risk_free_rate) / std_return
    mu = np.mean(portfolio_returns,axis=1)
    sigma = np.std(portfolio_returns,axis=1)
    # portfolio sharpe ratios shape: num_weight_sims x num_price_sims
    portfolio_sharpes = (mu-rfr)/sigma
    assert portfolio_sharpes.shape == (num_weight_sims,num_price_sims)

    # calculate the average sharpe for each weight sim and determine the optimal weights
    avg_sharpes = np.mean(portfolio_sharpes,axis=1)
    assert avg_sharpes.shape == (num_weight_sims,)
    optimal_weight_idx = np.argmax(avg_sharpes)
    # take the first vertex from the optimal weight matrix because these are duplicated across the dates
    assert np.all(weight_simulations[0] == weight_simulations[0][0])
    optimal_weights = weight_simulations[optimal_weight_idx][0]
    return optimal_weights

def monte_carlo(prices,num_sim_dates,num_price_sims,num_weight_sims):
    # aggregate function for monte carlo simulation for optimal weights
    log_returns = get_log_returns(prices)
    price_simulations = monte_carlo_price_simulations(prices,log_returns,num_sim_dates,num_price_sims)
    weight_simulations = monte_carlo_weights(price_simulations,num_weight_sims)
    optimal_weights = monte_carlo_optimal_weights(price_simulations,weight_simulations)
    return optimal_weights

# testing
# def main():
#     csv_file = 'stock_details_5_years.csv'
#     num_dates = 10
#     num_tickers = 10
#     num_sim_dates = 10
#     num_price_sims = 20
#     num_weight_sims = 30

#     close_data, dates, tickers = read_raw_csv(csv_file)
#     dates_subset = dates[:num_dates]
#     tickers_subset = tickers[:num_tickers]
#     close_subset = close_data[close_data['Date'].isin(dates_subset)]
#     close_subset = close_data[(close_data['Date'].isin(dates_subset)) & 
#                                 (close_data['Company'].isin(tickers_subset))]
#     prices = get_prices_matrix(close_subset,dates_subset,tickers_subset)

#     log_returns = get_log_returns(prices)
#     mu = get_mean_returns(log_returns)

#     price_simulations = monte_carlo_price_simulations(prices,log_returns,num_sim_dates,num_price_sims)
#     weight_simulations = monte_carlo_weights(price_simulations,num_weight_sims)

#     optimal_weights = monte_carlo_optimal_weights(price_simulations,weight_simulations)
#     print(optimal_weights)
#     print(np.sum(optimal_weights))
#     return 0


# if __name__ == "__main__":
#     main()