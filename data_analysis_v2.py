#%%
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from workalendar.asia import hong_kong
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

#%%
price_col_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Open(t+1)', 'Open(t+1) >= Close', 'Volume']

#convert bert to pca score
bert_df = pd.read_csv('bert_output20171001-20200430_v1.gzip', compression='gzip', index_col='title')
pca = PCA(n_components=.98)
pca.fit(bert_df)
print('number of pca columns:', pca.n_components_)
pca_cols = ['bert_pca{}'.format(1+i) for i in range(pca.n_components_)]
bert_pca_df = pd.DataFrame(pca.transform(bert_df), index=bert_df.index, columns=pca_cols)

news_df = pd.read_csv('data/news_20171001-20200430.gzip', compression='gzip', usecols=['title', 'date']) #for mapping
sentiment_df = pd.read_csv('normalised_sentiment_analysis_using_sia.csv', usecols=['title_pos', 'desc_pos', 'content_pos'])

sentiment_df['sum_mean'] = (sentiment_df['title_pos'] + sentiment_df['desc_pos'] + sentiment_df['content_pos'])/3
sentiment_cols = sentiment_df.columns.tolist()
news_df_merged = pd.concat([news_df, sentiment_df], axis=1)
news_df_merged = news_df_merged.set_index('title').merge(bert_pca_df, left_index=True, right_index=True, how='left')
news_df_merged_groupby_mean = news_df_merged.groupby('date')[pca_cols+sentiment_cols].agg(['mean'])
news_df_merged_groupby_sum = news_df_merged.groupby('date')[sentiment_cols].agg(['sum'])
news_df_merged_groupby = news_df_merged_groupby_mean.merge(news_df_merged_groupby_sum, left_index=True, right_index=True)
news_df_merged_groupby.columns = ['_'.join(col) for col in news_df_merged_groupby.columns]
news_df_merged_groupby['news_count'] = news_df_merged['date'].value_counts()

#create holiday date columns and mapping
cal = hong_kong.HongKong()
holiday_list = [str(day[0]) for day in cal.holidays(2018)+cal.holidays(2019)+cal.holidays(2020)]
news_df_merged_groupby['holiday'] = np.where(news_df_merged_groupby.index.isin(holiday_list), 1, 0)

price_700_df = pd.read_csv('data/price_20171001-20200430.gzip', compression='gzip', index_col='Date')
price_700_df['date'] = price_700_df.index
news_df_merged_groupby2 = news_df_merged_groupby.merge(price_700_df[['date']], left_index=True, right_index=True, how='left')

#fill the Sat, Sun and holiday for mapping
t = 1
while news_df_merged_groupby2['date'].isnull().any():
    news_df_merged_groupby2['date'] = news_df_merged_groupby2['date'].fillna(news_df_merged_groupby2['date'].shift(t))
    t += 1
news_df_merged_groupby3 = news_df_merged_groupby2.groupby(news_df_merged_groupby2['date'])[news_df_merged_groupby.columns.tolist()].sum()

price_700_df_merged = price_700_df.merge(news_df_merged_groupby3, left_index=True, right_index=True)

#shift the columns
shift_col_list = [col for col in price_700_df_merged.columns if col not in price_col_list+['date', 'holiday']]
shift_number = 4
for t in range(shift_number):
    price_700_df_merged[[col+'(t-{})'.format(t+1) for col in shift_col_list]] = price_700_df_merged[shift_col_list].shift(t+1)
price_700_df_merged = price_700_df_merged.iloc[shift_number:, ]

#create date related columns
price_700_df['date'] = pd.to_datetime(price_700_df.index, yearfirst=True)
price_700_df['month'] = price_700_df['date'].dt.month
price_700_df['day'] = price_700_df['date'].dt.day
price_700_df['weekday'] = price_700_df['date'].dt.weekday
# price_700_df_merged.head()

#%%
X_train, X_test, y_train, y_test = train_test_split(price_700_df_merged.drop(price_col_list+['date'], axis=1)
    , price_700_df_merged['Open(t+1) >= Close'], test_size=.2, random_state=1, stratify=price_700_df_merged['Open(t+1) >= Close'])
# data_len = price_700_df_merged.shape[0]
# X_train = price_700_df_merged.drop(price_col_list+['date'], axis=1).iloc[:9*data_len//10:, :]
# X_test = price_700_df_merged.drop(price_col_list+['date'], axis=1).iloc[9*data_len//10:, :]
# y_train = price_700_df_merged['Open(t+1) >= Close'].iloc[:9*data_len//10]
# y_test = price_700_df_merged['Open(t+1) >= Close'].iloc[9*data_len//10:]

print(X_train.shape)
for col in X_train:
    if X_train[col].isnull().sum() > 0:
        print('missing is found in:', col)
print('target rate', y_train.mean())

ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns, index=X_test.index)

n_jobs = 15
random_state_model = 1
np.random.seed(random_state_model) #seed random seed in numpy overall level

#%%
xgbc = XGBClassifier(random_state=random_state_model, n_jobs=n_jobs, eval_set=[(X_train, y_train)], eval_metric='auc', early_stopping_rounds=5)
param_dist_xgb = {'n_estimator': randint(10, 15)
            , 'max_depth': randint(1, 3)
            , 'learning_rate': uniform(.005, .01)
            , 'min_child_weight': uniform(.01, .1)
            , 'subsample': uniform(.3, .6)
            , 'lambda': uniform(2, 3) #l2
            , 'alpha': uniform(1, 3)} #l1
rs_xgb = RandomizedSearchCV(xgbc, param_dist_xgb, n_iter=20, return_train_score=True, n_jobs=n_jobs, scoring='roc_auc', cv=StratifiedShuffleSplit(n_splits=3, test_size=.2, random_state=random_state_model))
rs_xgb.fit(X_train, y_train)

cv_xgb = pd.DataFrame.from_dict(rs_xgb.cv_results_).sort_values('rank_test_score', ascending=False)
# print(cv_xgb[['mean_train_score', 'mean_test_score']])
print('Hightest cross valid score in XGB:\n'
    , 'Cross train score: {}'.format(cv_xgb.loc[cv_xgb['mean_test_score'].idxmax(), 'mean_train_score'])
    , 'Cross valid score: {}'.format(cv_xgb.loc[cv_xgb['mean_test_score'].idxmax(), 'mean_test_score'])
    , 'param: {}'.format(cv_xgb.loc[cv_xgb['mean_test_score'].idxmax(), 'params']))
importance_feature_df = pd.DataFrame({'col': X_train.columns, 'importance': rs_cv.best_estimator_.feature_importances_})
print(importance_feature_df.sort_values('importance', ascending=False).head(20))

#%%
param_dist_svc = {'kernel': ['sigmoid', 'rbf', 'poly'], 'C': uniform(3, 17)}
svc = SVC(probability=True, random_state=random_state_model)
rs_svc = RandomizedSearchCV(svc, param_dist_svc, n_iter=30
    , return_train_score=True, n_jobs=n_jobs, scoring='roc_auc', cv=StratifiedShuffleSplit(n_splits=3, test_size=.2, random_state=random_state_model))
rs_svc.fit(X_train, y_train)
cv_svc = pd.DataFrame.from_dict(rs_svc.cv_results_).sort_values('rank_test_score', ascending=False)
print('Hightest cross valid score in SVC:\n'
    , 'Cross train score: {}'.format(cv_svc.loc[cv_svc['mean_test_score'].idxmax(), 'mean_train_score'])
    , 'Cross valid score: {}'.format(cv_svc.loc[cv_svc['mean_test_score'].idxmax(), 'mean_test_score'])
    , 'param: {}'.format(cv_svc.loc[cv_svc['mean_test_score'].idxmax(), 'params']))

#%%
vc = VotingClassifier(estimators=[('xgbc', xgbc), ('svc', svc)], voting='soft', n_jobs=n_jobs)
param_dist_vc = {}
for k, v in param_dist_xgb.items():
    param_dist_vc['xgbc__'+k] = v
for k, v in param_dist_svc.items():
    param_dist_vc['svc__'+k] = v

rs_vc = RandomizedSearchCV(vc, param_dist_vc, n_iter=40
    , return_train_score=True, n_jobs=n_jobs, scoring='roc_auc', cv=StratifiedShuffleSplit(n_splits=3, test_size=.2, random_state=random_state_model))
rs_vc.fit(X_train, y_train)
cv_vc = pd.DataFrame.from_dict(rs_vc.cv_results_).sort_values('rank_test_score', ascending=False)
print('Hightest cross valid score in Voting Classifier:\n'
    , 'Cross train score: {}'.format(cv_vc.loc[cv_vc['mean_test_score'].idxmax(), 'mean_train_score'])
    , 'Cross valid score: {}'.format(cv_vc.loc[cv_vc['mean_test_score'].idxmax(), 'mean_test_score'])
    , 'param: {}'.format(cv_vc.loc[cv_vc['mean_test_score'].idxmax(), 'params']))

#%%
best_alg = rs_xgb
y_train_p = best_alg.best_estimator_.predict_proba(X_train)[:,  1]
y_test_p = best_alg.best_estimator_.predict_proba(X_test)[:, 1]
print('roc_auc in train:', roc_auc_score(y_train, y_train_p))
print('roc_auc in test:', roc_auc_score(y_test, y_test_p))
print('accuracy in train:', accuracy_score(y_train, np.where( y_train_p >= .5, 1, 0)))
print('accuracy in test:', accuracy_score(y_test, np.where( y_test_p >= .5, 1, 0)))

#%%
#test the xgboost without bert and setiment analysis info
non_news_cols = [col for col in X_train if all([s not in col for s in ['bert_', 'sum_mean', 'title_pos', 'desc_pos', 'content_pos', 'news_count']])]
X_train_no_news = X_train[non_news_cols]
X_test_no_news = X_test[non_news_cols]
rs_xgb_no_news = RandomizedSearchCV(xgbc, param_dist_xgb, n_iter=20, return_train_score=True, n_jobs=n_jobs, scoring='roc_auc', cv=StratifiedShuffleSplit(n_splits=3, test_size=.2, random_state=random_state_model))
rs_xgb_no_news.fit(X_train_no_news, y_train)

cv_xgb_no_news = pd.DataFrame.from_dict(rs_xgb_no_news.cv_results_).sort_values('rank_test_score', ascending=False)
# print(cv_xgb_no_news[['mean_train_score', 'mean_test_score']])
print('Hightest cross valid score in XGB without the news:\n'
    , 'Train: {}'.format(cv_xgb_no_news.loc[cv_xgb_no_news['mean_test_score'].idxmax(), 'mean_train_score'])
    , 'Test: {}'.format(cv_xgb_no_news.loc[cv_xgb_no_news['mean_test_score'].idxmax(), 'mean_test_score'])
    , 'param: {}'.format(cv_xgb_no_news.loc[cv_xgb_no_news['mean_test_score'].idxmax(), 'params']))
importance_feature_df_no_news = pd.DataFrame({'col': X_train_no_news.columns, 'importance': rs_xgb_no_news.best_estimator_.feature_importances_})
print(importance_feature_df_no_news.sort_values('importance', ascending=False).head(10))

#%%
best_alg = rs_xgb_no_news
y_train_p = best_alg.best_estimator_.predict_proba(X_train_no_news)[:,  1]
y_test_p = best_alg.best_estimator_.predict_proba(X_test_no_news)[:, 1]
print('roc_auc in train:', roc_auc_score(y_train, y_train_p))
print('roc_auc in test:', roc_auc_score(y_test, y_test_p))
print('accuracy in train:', accuracy_score(y_train, np.where( y_train_p >= .5, 1, 0)))
print('accuracy in test:', accuracy_score(y_test, np.where( y_test_p >= .5, 1, 0)))

#======================================================================
#======================================================================
#======================================================================

#%%
lag = 10
X_lstm = price_700_df_merged.drop([col for col in price_700_df_merged if '(t-' in col]+['Open(t+1) >= Close', 'Open(t+1)', 'date'], axis=1).copy()
input_dim, input_col = X_lstm.shape[0], X_lstm.shape[1]
minmax = MinMaxScaler(feature_range=(0, 1))
X_lstm = minmax.fit_transform(X_lstm)

X_lstm_list, y_lstm_list = [], []
for i in range(lag, len(X_lstm)):
    X_lstm_list.append(X_lstm[i-lag:i, :])
    y_lstm_list.append(np.array(price_700_df_merged['Open(t+1) >= Close'].iloc[i]))

X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(np.array(X_lstm_list)
    , np.array(y_lstm_list), test_size=.1, random_state=3, stratify=np.array(y_lstm_list))
print('X lstm train shape', X_lstm_train.shape)
print('y lstm train shape', y_lstm_train.shape)

#lstm model
lstm_model = Sequential()
# lstm_model.add(LSTM(units=64, dropout=.5, recurrent_dropout=.5, return_sequences=True))
# lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=10, dropout=.5, recurrent_dropout=.5))
lstm_model.add(BatchNormalization())
# lstm_model.add(Dense(10, activation='elu'))
# lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
history = lstm_model.fit(X_lstm_train, y_lstm_train, epochs=500, verbose=2, validation_data=[X_lstm_test, y_lstm_test], callbacks=[early_stopping], batch_size=32)
y_lstm_train_p = lstm_model.predict(X_lstm_train)
y_lstm_test_p = lstm_model.predict(X_lstm_test)

print('roc_auc in train:', roc_auc_score(y_lstm_train, y_lstm_train_p))
print('roc_auc in test:', roc_auc_score(y_lstm_test, y_lstm_test_p))
print('accuracy in train:', accuracy_score(y_lstm_train, np.where( y_lstm_train_p >= .5, 1, 0)))
print('accuracy in test:', accuracy_score(y_lstm_test, np.where( y_lstm_test_p >= .5, 1, 0)))

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('number of epoch')
plt.ylabel('loss (binary cross entropy)')
plt.title('Training LSTM model')
plt.legend()
plt.show()
