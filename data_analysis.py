#%%
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from workalendar.asia import hong_kong
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedShuffleSplit
# from sklearn.metrics import 
from xgboost import XGBClassifier

price_col_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Open(t+1)', 'Open(t+1) >= Close', 'Volume']

bert_df = pd.read_csv('bert_output.gzip', compression='gzip', index_col='title')
pca = PCA(n_components=.95)
pca.fit(bert_df)
print('number of pca columns:', pca.n_components_)
pca_cols = ['title_pca{}'.format(1+i) for i in range(pca.n_components_)]
title_pca_df = pd.DataFrame(pca.transform(bert_df), index=bert_df.index, columns=pca_cols)
#%%
news_df = pd.read_csv('data/news_20171001-20200430.gzip', compression='gzip', index_col='title')
news_df_merged = news_df.merge(title_pca_df, left_index=True, right_index=True, how='left')
news_df_merged_groupby = news_df_merged.groupby('date')[pca_cols].sum()
# news_df_merged_groupby.columns = ['_'.join(col) for col in news_df_merged_groupby.columns]
news_df_merged_groupby['news_count'] = news_df_merged['date'].value_counts()

#%%
#create holiday column
cal = hong_kong.HongKong()
holiday_list = [str(day[0]) for day in cal.holidays(2018)+cal.holidays(2019)+cal.holidays(2020)]
news_df_merged_groupby['holiday'] = np.where(news_df_merged_groupby.index.isin(holiday_list), 1, 0)

price_700_df = pd.read_csv('data/price_20171001-20200430.gzip', compression='gzip', index_col='Date')
price_700_df['date'] = price_700_df.index
news_df_merged_groupby2 = news_df_merged_groupby.merge(price_700_df[['date']], left_index=True, right_index=True, how='left')

#%%
t = 1
while news_df_merged_groupby2['date'].isnull().any():
    news_df_merged_groupby2['date'] = news_df_merged_groupby2['date'].fillna(news_df_merged_groupby2['date'].shift(t))
    t += 1

news_df_merged_groupby3 = news_df_merged_groupby2.groupby(news_df_merged_groupby2['date'])[pca_cols+['news_count', 'holiday']].sum()

price_700_df_merged = price_700_df.merge(news_df_merged_groupby3, left_index=True, right_index=True)

shift_col_list = [col for col in price_700_df_merged.columns if col not in price_col_list+['date', 'holiday']]
shift_number = 5
for t in range(shift_number):
    price_700_df_merged[[col+'(t+{})'.format(t+1) for col in shift_col_list]] = price_700_df_merged[shift_col_list].shift(t+1)
price_700_df_merged = price_700_df_merged.iloc[shift_number:, ]


#create date related columns
price_700_df['date'] = pd.to_datetime(price_700_df.index, yearfirst=True)
price_700_df['month'] = price_700_df['date'].dt.month
price_700_df['day'] = price_700_df['date'].dt.day
price_700_df['weekday'] = price_700_df['date'].dt.weekday

price_700_df_merged.head()

#%%
X_train, X_test, y_train, y_test = train_test_split(price_700_df.drop(price_col_list+['date'], axis=1)
    , price_700_df['Open(t+1) >= Close'], test_size=.2, random_state=1, stratify=price_700_df['Open(t+1) >= Close'])
print(X_train.shape)
for col in X_train:
    if X_train[col].isnull().sum() > 0:
        print(col)
print('target rate', y_train.mean())

#%%
random_state_model = 1
n_jobs = 15
xgbc = XGBClassifier(random_state=random_state_model, n_jobs=n_jobs, eval_set=[(X_train, y_train)], eval_metric='auc', early_stopping_rounds=5)
param_dist = {'n_estimator': randint(10, 20)
            , 'max_depth': randint(1, 2)
            , 'learning_rate': uniform(.005, .01)
            , 'min_child_weight': uniform(.1, .015)
            , 'subsample': uniform(.3, .2)
            , 'lambda': uniform(2, 3) #l2
            , 'alpha': uniform(1, 3)} #l1
rs_cv = RandomizedSearchCV(xgbc, param_dist, n_iter=10
    , return_train_score=True, n_jobs=n_jobs, scoring='roc_auc', cv=StratifiedShuffleSplit(n_splits=3, test_size=.2, random_state=random_state_model))
rs_cv.fit(X_train, y_train)

cv_result = pd.DataFrame.from_dict(rs_cv.cv_results_).sort_values('rank_test_score', ascending=False)
print(cv_result[['mean_train_score', 'mean_test_score']])

importance_feature_df = pd.DataFrame({'col': X_train.columns, 'importance': rs_cv.best_estimator_.feature_importances_})
print(importance_feature_df.sort_values('importance', ascending=False).head(30))

# %%
