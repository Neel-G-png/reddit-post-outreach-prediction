import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
from VADdet import analyze
import pickle
import time

ml_model = joblib.load("savedmodels/mlp_pickle_model12.joblib")


path = "files/RED_DATA_TRAIN.csv"

data = pd.read_csv(path)
data_removed = data.drop(['redditor','type','text','proc_text','proc_title','genre','absolute_words_ratio','neg_log_prob_aw_ratio'],axis = 1)

data_removed = data_removed.dropna(subset = ['title','subreddit','datetime','valence','arousal','dominance'])

train_x ,train_y = data_removed.drop('score',axis = 1), data_removed[['score']]

tfidf_subreddit = TfidfVectorizer(ngram_range=(1, 1), max_features=None)
subreddit_sparse = tfidf_subreddit.fit_transform(train_x['subreddit'])

tfidf_title = TfidfVectorizer(ngram_range=(1, 5), max_features=None)
title_sparse = tfidf_title.fit_transform(train_x['title'])
date_time = train_x[['datetime']]

stacked_val = train_x.drop(['title','subreddit'],axis = 1)
scaler = StandardScaler()
scaled_date = scaler.fit_transform(stacked_val)

start = time.time()
subreddit=["books"]

title = "I love mein kampf "
mode = 'mean'
V,A,D = analyze(title,mode)
res = (V + A + D)/3
print(f"\nVAD = {V}\t{A}\t{D}")

print("\nRESULT = ",res)

DVAD = [[1499858857,V/10,A/10,D/10]]


sub_sparse = tfidf_subreddit.transform(subreddit)
tit_sparse = tfidf_title.transform([title])
test_DVAD = scaler.transform(DVAD)
test_x_sparse = hstack([tit_sparse, sub_sparse, test_DVAD])
pred_y = ml_model.predict(test_x_sparse)

print(pred_y)
print(f"\nTOTAL TIME TAKEN {time.time()-start}")
