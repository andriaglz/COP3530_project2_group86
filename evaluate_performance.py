from data_preprocessing import *
from markowitz import *
from monte_carlo import *
import time
import tracemalloc

def calc_sharpe(weights, prices, rfr=0):
    # log returns shape: num_dates-1 x num_tickers
    log_returns = get_log_returns(prices)
    _, num_tickers = log_returns.shape
    # arithmetic returns shape: num_dates-1 x num_tickers
    arithmetic_returns = np.exp(log_returns) - 1

    # weights shape: num_tickers
    assert weights.shape == (num_tickers,)

    # calculate portfolio returns
    portfolio_returns = arithmetic_returns @ weights

    # calculate sharpe
    # sharpe = (mu-rfr)/sigma = (average_return - risk_free_rate) / std_return
    mu = np.mean(portfolio_returns, axis=0)
    rfr_deannualized = (1.0 + rfr) ** (1.0 / 252) - 1.0
    sigma = np.std(portfolio_returns, ddof=1)
    sharpe = (mu - rfr_deannualized) / sigma
    annualized_sharpe = sharpe * np.sqrt(252)

    return annualized_sharpe


def get_inputs(close_data, dates, selected_tickers):
    prices = get_prices_matrix(close_data, dates, selected_tickers)

    tracemalloc.start()
    curr = time.time()
    markowitz_weights = markowitz(prices)
    marko_time = time.time() - curr
    memory_tuple = tracemalloc.get_traced_memory()
    marko_memory = memory_tuple[1] - memory_tuple[0];
    tracemalloc.stop()

    tracemalloc.start()
    curr = time.time()
    monte_carlo_weights = monte_carlo(prices=prices,
                                      num_price_sims=100, num_sim_dates=100, num_weight_sims=100)
    monte_time = time.time() - curr
    memory_tuple = tracemalloc.get_traced_memory()
    monte_memory = memory_tuple[1] - memory_tuple[0];
    tracemalloc.stop()
    monte_carlo_sharpe = calc_sharpe(monte_carlo_weights, prices)
    markowitz_sharpe = calc_sharpe(markowitz_weights, prices)

    return {'Tickers': selected_tickers,
            'Markowitz Weights': [round(w, 4) for w in markowitz_weights],
            'Monte Carlo Weights': [round(w, 4) for w in monte_carlo_weights],
            'Markowitz Sharpe': round(markowitz_sharpe, 4),
            'Monte Carlo Sharpe': round(monte_carlo_sharpe, 4),
            'Markowitz Time': round(marko_time, 4),
            'Monte Carlo Time': round(monte_time, 4),
            'Markowitz Memory': round(marko_memory, 4),
            'Monte Carlo Memory': round(monte_memory, 4)}

# # testing
# def main():
#     csv_file = 'stock_details_5_years.csv'
#     # for data subset
#     num_dates = 1200
#     num_tickers = 400
#     # for monte carlo simulation
#     num_sim_dates = 100
#     num_price_sims = 100
#     num_weight_sims = 100

#     # preprocessing
#     close_data, dates, tickers = read_raw_csv(csv_file)
#     # print("Tickers to choose from:\n",tickers)
#     dates_subset = dates[:num_dates]
#     # tickers_subset = tickers[:num_tickers]
#     tickers_subset = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA']
#     # tickers_subset = ['AMZN','NVDA','META']
#     # tickers_subset = ['NVDA']
#     print(tickers_subset)

#     close_subset = close_data[close_data['Date'].isin(dates_subset)]
#     close_subset = close_data[(close_data['Date'].isin(dates_subset)) &
#                                 (close_data['Company'].isin(tickers_subset))]
#     prices = get_prices_matrix(close_subset,dates_subset,tickers_subset)

#     # portfolio optimization weights
#     markowitz_weights = markowitz(prices)
#     monte_carlo_weights = monte_carlo(prices,num_sim_dates,num_price_sims,num_weight_sims)
#     print("Monte Carlo Weights:",monte_carlo_weights)
#     print("Markowitz Sharpe Weights:",markowitz_weights)
#     # print(np.sum(monte_carlo_weights),np.sum(markowitz_weights))

#     # evaluation
#     monte_carlo_sharpe = calc_sharpe(monte_carlo_weights,prices)
#     markowitz_sharpe = calc_sharpe(markowitz_weights,prices)
#     print("Monte Carlo Sharpe Ratio:",monte_carlo_sharpe)
#     print("Markowitz Sharpe Ratio:",markowitz_sharpe)


#     return 0


# if __name__ == "__main__":
#     main()
