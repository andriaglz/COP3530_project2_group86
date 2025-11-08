import pandas as pd
import numpy as np

def read_raw_csv(csv_file):
    '''
    Reads in the raw data from the Kaggle dataframe, 
        extracts relevant columns, and removes tickers with missing dates
    
    Input: string for csv file path
    Output:
        close_data - pandas dataframe for close data (Date | Close | Company)
        dates - np.array of dates present in the final dataframe
        tickers - np.array of tickers present in the final dataframe

    Additional Information
    - Raw data
        Date | Open | High | Low | Close | Volume | ... | Company
        shape: 602,963 x 8
    - Close data
        Date | Close | Company
        shape: num_dates x num_tickers
    '''
    raw_data = pd.read_csv(csv_file)
    required_columns = ['Date', 'Close', 'Company']
    assert(all(col in raw_data.columns for col in required_columns))
    close_data = raw_data[required_columns]

    # extract dates and tickers
    dates = np.array(close_data['Date'].unique())
    num_dates = dates.size

    # remove companies with not enough data
    company_counts = close_data['Company'].value_counts()
    valid_companies = company_counts[company_counts==num_dates].index.tolist()
    close_data = close_data[close_data['Company'].isin(valid_companies)]
    tickers = np.array(close_data['Company'].unique())

    return close_data, dates, tickers

def get_prices_matrix(close_data,dates,tickers):
    '''
    Creates a matrix of price data for each ticker where
        the rows represent each date and the columns represent each ticker
    
    Input:
        close_data - pandas dataframe for close data (Date | Close | Company)
        dates - np.array of dates present in the final dataframe
        tickers - np.array of tickers present in the final dataframe
    Output:
        prices - np.array with the prices for each ticker 
            (shape: num_dates x num_tickers)
    '''
    # extract number of dates and tickers
    num_dates = len(dates)
    num_tickers = len(tickers)

    # create array of prices (rows represent dates, columns represent tickers)
    prices = np.zeros(num_dates*num_tickers)
    for i in range(num_tickers):
        ticker_prices = np.array(close_data[close_data['Company']==tickers[i]]['Close'])
        assert(ticker_prices.size==num_dates)
        start_idx = i*num_dates
        end_idx = start_idx+num_dates
        prices[start_idx:end_idx] = ticker_prices
    prices = prices.reshape(num_dates,num_tickers)

    return prices

def get_log_returns(prices):
    # calculate returns
    # log_returns = ln(price / prev_price)
    # shape: (num_dates-1) x num_tickers
    return np.log(prices[1:]/prices[:-1])

def get_mean_returns(log_returns):
    # mu = 1/num_dates * sum(1,num_dates){log_return_i}
    # shape: num_tickers x 1
    return np.mean(log_returns,axis=0)

