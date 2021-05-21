import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import r2_score,accuracy_score
import joblib
import numpy as np
from VADdet import analyze

path = "../files/new_train_test/new_Train_data.csv"

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

train_x_sparse = hstack([title_sparse,subreddit_sparse,scaled_val])
print("\t ############ TRAINING MODEL ############")
# train_y = train_y.astype('int')

ml_model = MLPClassifier(max_iter=45,hidden_layer_sizes = (35,),verbose = True) 
           #Ridge(alpha = 0.0001)
           #MLPClassifier(max_iter=30,hidden_layer_sizes = (25,5,),verbose = True)
           #LinearRegression()
ml_model.fit(train_x_sparse,y.values.ravel())
joblib.dump(ml_model, 'mlp_hour.joblib')

print(ml_model.score(train_x_sparse, y)) 

print("\t ############ TRAINING COMPLETE ############")

# ml_model = joblib.load("mlp_hour.joblib")
######################### TESTING #########################
test_data = pd.read_csv("../files/new_train_test/finaltest_data.csv")

test_data_removed = test_data.drop(['redditor','type','text','proc_text','proc_title','genre','absolute_words_ratio','neg_log_prob_aw_ratio'],axis = 1)

test_data_removed = test_data_removed.dropna(subset = ['title','subreddit','datetime','valence','arousal','dominance','hour'])

test_x, test_y = test_data_removed.drop('score',axis = 1), test_data_removed[['score']]

sub_sparse = tfidf_subreddit.transform(test_x['subreddit'])
tit_sparse = tfidf_title.transform(test_x['title'])

test_date_time = test_x[['hour']]
test_valence = test_x[['valence']]
test_arousal = test_x[['arousal']]
test_dominance = test_x[['dominance']]
test_date = scaler.transform(test_date_time)
# print(test_date)
test_scaled_val = np.hstack([test_date,test_valence,test_arousal,test_dominance])

test_x_sparse = hstack([tit_sparse, sub_sparse, test_scaled_val])

pred_y = ml_model.predict(test_x_sparse)
print(pred_y)

