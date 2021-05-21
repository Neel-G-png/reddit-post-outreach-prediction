import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import r2_score,accuracy_score
import joblib
import numpy as np

path = "files/RED_DATA_TRAIN.csv"

data = pd.read_csv(path)
data_removed = data.drop(['redditor','type','text','proc_text','proc_title','genre','absolute_words_ratio','neg_log_prob_aw_ratio'],axis = 1)

# print(data_removed['subreddit'].isna().sum())
data_removed = data_removed.dropna(subset = ['title','subreddit','datetime','valence','arousal','dominance'])
# data_removed = data_removed.dropna(subset = ['subreddit'])

train_x ,train_y = data_removed.drop('score',axis = 1), data_removed[['score']]

tfidf_subreddit = TfidfVectorizer(ngram_range=(1, 1), max_features=None)
subreddit_sparse = tfidf_subreddit.fit_transform(train_x['subreddit'])

tfidf_title = TfidfVectorizer(ngram_range=(1, 5), max_features=None)
title_sparse = tfidf_title.fit_transform(train_x['title'])

date_time = train_x[['datetime']]
valence = train_x[['valence']]
arousal = train_x[['arousal']]
dominance = train_x[['dominance']]


scaler = StandardScaler()
scaled_date = scaler.fit_transform(date_time)
scaled_val = np.hstack([scaled_date,valence,arousal,dominance])

print("\nDATE SCALED !!\n",scaled_val)
train_x_sparse = hstack([title_sparse,subreddit_sparse,scaled_val])
print("\t ############ TRAINING MODEL ############")

ml_model = MLPClassifier(max_iter=35,hidden_layer_sizes = (35,),verbose = True) 
ml_model.fit(train_x_sparse,train_y.values.ravel())
joblib.dump(ml_model, 'savedmodels/mlp_pickle_model12.joblib')

print(ml_model.score(train_x_sparse, train_y)) 

print("\t ############ TRAINING COMPLETE ############")

test_data = pd.read_csv("files/RED_DATA_TEST.csv")

test_data_removed = test_data.drop(['redditor','type','text','proc_text','proc_title','genre','absolute_words_ratio','neg_log_prob_aw_ratio'],axis = 1)

test_data_removed = test_data_removed.dropna(subset = ['title','subreddit','datetime','valence','arousal','dominance'])

test_x, test_y = test_data_removed.drop('score',axis = 1), test_data_removed[['score']]

sub_sparse = tfidf_subreddit.transform(test_x['subreddit'])
tit_sparse = tfidf_title.transform(test_x['title'])
test_date_time = test_x[['datetime']]
test_valence = test_x[['valence']]
test_arousal = test_x[['arousal']]
test_dominance = test_x[['dominance']]
test_date = scaler.transform(test_date_time)
test_scaled_val = np.hstack([test_date,test_valence,test_arousal,test_dominance])

test_x_sparse = hstack([tit_sparse, sub_sparse, test_scaled_val])

pred_y = ml_model.predict(test_x_sparse)
print(pred_y)

