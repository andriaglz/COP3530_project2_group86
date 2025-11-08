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

