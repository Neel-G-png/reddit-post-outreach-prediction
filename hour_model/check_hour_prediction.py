
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import r2_score,accuracy_score
import joblib
from VADdet import analyze
import numpy as np

path = "../files/new_train_test/new_tTrain_data.csv"

data = pd.read_csv(path)
calculated = pd.DataFrame(columns=['actual_score','hour_pred'])
data_removed = data.drop(['redditor','type','text','proc_text','proc_title','genre','absolute_words_ratio','neg_log_prob_aw_ratio'],axis = 1)

data_removed = data_removed.dropna(subset = ['title','subreddit','datetime','valence','arousal','dominance','hour'])


train_x ,y = data_removed.drop('score',axis = 1), data_removed[['score']]

tfidf_subreddit = TfidfVectorizer(ngram_range=(1, 1), max_features=None)
subreddit_sparse = tfidf_subreddit.fit_transform(train_x['subreddit'])


#changing ngram range 
tfidf_title = TfidfVectorizer(ngram_range=(2, 5), max_features=None)
title_sparse = tfidf_title.fit_transform(train_x['title'])

hour = train_x[['hour']]
valence = train_x[['valence']]
arousal = train_x[['arousal']]
dominance = train_x[['dominance']]

scaler = StandardScaler()
scaled_date = scaler.fit_transform(hour)
scaled_val = np.hstack([scaled_date,valence,arousal,dominance])

ml_model = joblib.load("mlp_hour.joblib")

hour_clock = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
time = [[10]]
subredddit = ["books"]
title = "Pizza Hut's BOOK IT! summer reading camp is back and we have so much nostalgia"
mode = "mean"
V = A = D = 0
V,A,D = analyze(title,mode)
V/=10
A/=10
D/=10

sub_sparse = tfidf_subreddit.transform(subredddit)
tit_sparse = tfidf_title.transform([title])
time_sparse = scaler.transform(time)

pred_sparse = hstack([tit_sparse,sub_sparse,time_sparse])
result = ml_model.predict(pred_sparse)
for hour in hour_clock:
    if hour != time:
        time_sparse = scaler.transform([[hour]])
        vad_sparse = np.hstack([time_sparse,[[V]],[[A]],[[D]]])
        pred_sparse = hstack([tit_sparse,sub_sparse,vad_sparse])
        test_res = ml_model.predict(pred_sparse)
        if test_res>result:
            print(f"\nYOU WOULD GET {test_res} UPVOTES IF YOU POSTED AT {hour}")

print(result)
