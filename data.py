
# %%
import os
import numpy as np
import pandas as pd
from newsapi import NewsApiClient
import yfinance as yf
import yahoofinancials

def newsapi_to_df(news_dict_list):
  for news in news_dict_list:
    news['from'] = news['source']['name']
  news_df = pd.DataFrame.from_dict(news_dict_list)
  news_df.drop(['source', 'urlToImage'], axis=1, inplace=True)
  news_df['date'] = [date[:10] for date in news_df['publishedAt']]
  news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')#in datetime format
  return news_df

def rsi(price, n=14):
    ''' rsi indicator '''
    gain = (price-price.shift(1)).fillna(0) # calculate price gain with previous day, first row nan is filled with 0

    def rsiCalc(p):
        # subfunction for calculating rsi for one lookback period
        avgGain = p[p>0].sum()/n
        avgLoss = -p[p<0].sum()/n 
        rs = avgGain/avgLoss
        return 100 - 100/(1+rs)

    # run for all periods with rolling_apply
    return gain.rolling(n).apply(rsiCalc) 

# %%
first_days = 14
price_700_df = yf.download('0700.HK', 
                      start='2018-03-29', 
                      end='2020-04-21', 
                      progress=False)
price_700_df['Open(t+1)'] = price_700_df['Open'].shift(-1)
price_700_df['rsi'] = rsi(price_700_df['Open'], n=first_days)
price_700_df = price_700_df.iloc[first_days:]
price_700_df['Open(t+1) >= Close'] = np.where(price_700_df['Open(t+1)'] >= price_700_df['Close'] , 1, 0)
price_700_df.head()
price_700_df.to_csv('data/price_20180423-20200421.gzip', compression='gzip')

# %%
newsapi = NewsApiClient(api_key='5209c394a0274f459880a2bd85e07e13') # Init
news_param = {'from_param': '2020-03-22', 'to': '2020-04-22', 'language': 'en', 'sort_by': 'popularity'} #param for getting news

news_df = pd.DataFrame()
for key_word in ['0700', 'tencent', '700.HK']:
  news_dict = newsapi.get_everything(q=key_word, **news_param)
  print('Total result by {}: {}'.format(key_word, news_dict['totalResults']))
  news_sub_df = newsapi_to_df(news_dict['articles'])
  print(key_word, news_sub_df.shape)
  news_df = news_df.append(news_sub_df, ignore_index=True)
print(news_df.shape)
news_df2 = news_df.drop_duplicates()
print(news_df2.shape) #no duplicates news from different searching words
display(news_df2.head())
news_df2.to_csv('data/news_20200322_20200421.gzip', compression='gzip')
