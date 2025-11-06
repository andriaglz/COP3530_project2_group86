from data_preprocessing import *
from markowitz import *
from monte_carlo import *

def calc_sharpe(weights,prices,rfr=0):
    # log returns shape: num_dates-1 x num_tickers
    log_returns = get_log_returns(prices)
    _,num_tickers = log_returns.shape
    # weights shape: num_tickers
    # print(weights.shape,num_tickers)
    assert weights.shape == (num_tickers,)

    # calculate portfolio returns
    individual_returns = log_returns * weights
    portfolio_returns = np.sum(individual_returns,axis=-1)

    # calculate sharpe
    # sharpe = (mu-rfr)/sigma = (average_return - risk_free_rate) / std_return
    mu = np.mean(portfolio_returns,axis=0)
    sigma = np.std(portfolio_returns,axis=0)
    sharpe = (mu-rfr)/sigma

    return sharpe


# testing
def main():
    csv_file = 'stock_details_5_years.csv'
    # for data subset
    num_dates = 1200
    # num_tickers = 10
    # for monte carlo simulation
    num_sim_dates = 100
    num_price_sims = 100
    num_weight_sims = 100

    # preprocessing
    close_data, dates, tickers = read_raw_csv(csv_file)
    # print("Tickers to choose from:\n",tickers)
    dates_subset = dates[:num_dates]
    # tickers_subset = tickers[:num_tickers]
    tickers_subset = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA']
    close_subset = close_data[close_data['Date'].isin(dates_subset)]
    close_subset = close_data[(close_data['Date'].isin(dates_subset)) & 
                                (close_data['Company'].isin(tickers_subset))]
    prices = get_prices_matrix(close_subset,dates_subset,tickers_subset)

    # portfolio optimization weights
    markowitz_weights = markowitz(prices)
    monte_carlo_weights = monte_carlo(prices,num_sim_dates,num_price_sims,num_weight_sims)
    print("Monte Carlo Weights:",monte_carlo_weights)
    print("Markowitz Sharpe Weights:",markowitz_weights)
    # print(np.sum(monte_carlo_weights),np.sum(markowitz_weights))

    # evaluation
    monte_carlo_sharpe = calc_sharpe(monte_carlo_weights,prices)
    markowitz_sharpe = calc_sharpe(markowitz_weights,prices)
    print("Monte Carlo Sharpe Ratio:",monte_carlo_sharpe)
    print("Markowitz Sharpe Ratio:",markowitz_sharpe)
    

    return 0


if __name__ == "__main__":
    main()