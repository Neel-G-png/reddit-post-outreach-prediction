# import nltk
# nltk.download('punkt')
import string
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
import pandas as pd

stop_words = set(stopwords.words('english'))

df = pd.read_csv("files/RED_DATA.csv")

# test_data = df.head(51088)
# test_data.to_csv("files/RED_DATA_TRAIN.csv",index = False)

for ind in range(63912):
    if ind%1000 == 0:
        print(ind/1000)
    sentence = " "
    text = str(df['title'][ind])
    word_tokens = word_tokenize(text)
    line_list = [w for w in word_tokens if not w in stop_words]
    text = sentence.join(line_list)
    df['title'][ind] = text

data = df.dropna(subset = ['title'])

data.to_csv('files/RED_DATA.csv',index = False)