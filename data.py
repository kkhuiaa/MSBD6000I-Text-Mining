
# %%
import os
import numpy as np
import pandas as pd
from newsapi import NewsApiClient
import yfinance as yf
import yahoofinancials
import ta

def newsapi_to_df(news_dict_list):
  if len(news_dict_list) > 0:
    for news in news_dict_list:
      news['from'] = news['source']['name']
    news_df = pd.DataFrame.from_dict(news_dict_list)
    news_df.drop(['source', 'urlToImage'], axis=1, inplace=True)
    news_df['date'] = [date[:10] for date in news_df['publishedAt']]
    news_df['date'] = pd.to_datetime(news_df['date'], format='%Y-%m-%d')#in datetime format
    return news_df
  else:
    return pd.DataFrame()

# def rsi(price, n=14):
#     ''' rsi indicator '''
#     gain = (price-price.shift(1)).fillna(0) # calculate price gain with previous day, first row nan is filled with 0

#     def rsiCalc(p):
#         # subfunction for calculating rsi for one lookback period
#         avgGain = p[p>0].sum()/n
#         avgLoss = -p[p<0].sum()/n 
#         rs = avgGain/avgLoss
#         return 100 - 100/(1+rs)

#     # run for all periods with rolling_apply
#     return gain.rolling(n).apply(rsiCalc) 

# %%
price_700_df = yf.download('0700.HK', 
                      start='2017-10-01', 
                      end='2020-04-30', 
                      progress=False)
price_700_df = ta.add_all_ta_features(price_700_df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
price_700_df['Open(t+1)'] = price_700_df['Open'].shift(-1)
price_700_df['Open(t+1) >= Close'] = np.where(price_700_df['Open(t+1)'] >= price_700_df['Close'] , 1, 0)
price_700_df.to_csv('data/price_20171001-20200430.gzip', compression='gzip')

#%%
newsapi = NewsApiClient(api_key='5209c394a0274f459880a2bd85e07e13') # Init
news_df = pd.DataFrame()
date_range = [str(date)[:10] for date in pd.date_range(start='10/1/2017', end='4/30/2020')] #, end='21/4/2020')
for date in date_range:
  df1 = newsapi_to_df(newsapi.get_everything(q='tencent 700', from_param=date, to=date, language='en', sort_by='relevancy', page_size=45)['articles']) #stronger criterion
  df2 = newsapi_to_df(newsapi.get_everything(q='tencent', from_param=date, to=date, language='en', sort_by='relevancy', page_size=5)['articles']) #weeker criterion
  df = df1.append(df2)
  print(date+' total news:', df.shape)
  news_df = news_df.append(df)
  print('='*20)
news_df = news_df.drop_duplicates(['title'])
print('final size of the text data', news_df.shape)
news_df.reset_index(drop=True).to_csv('data/news_20171001-20200430.gzip', compression='gzip')
