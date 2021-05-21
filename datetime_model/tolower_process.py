import pandas as pd
import re
from datetime import datetime,timezone

#datetime , score , title , subreddit

path = "cleaned_reddit_data.csv"
df = pd.read_csv("reddit_data.csv")


save_this = pd.read_csv(path)
#data = data.drop(['label','author','date','created_utc','score','ups','downs','subreddit'], axis=1)
# df = data.dropna(subset=['title'])
df['subreddit']=df['subreddit'].str.lower()
df['title']=df['title'].str.lower()
print("\t\t\n\n!!! ----- DONE lower ----- !!!")
data_index = df.index
for ind in range(228599):#data_index:
    if ind%1000==0:
        print(ind/1000)
    text = str(df['title'][ind])
    time_date = df['datetime'][ind]
    pattern = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"[^A-Za-z0-9_, ]+", "", text)
    # text = re.sub(r"[’]", "'",text)
    # text = text.replaceAll("\uFFFD", "")
    df['title'][ind] = text
    utc_date = datetime.strptime(time_date, '%Y-%m-%d %H:%M:%S')
    timestamp = utc_date.replace(tzinfo=timezone.utc).timestamp()
    df['datetime'][ind] = timestamp
save_this['datetime'] = df['datetime']
save_this['title'] = df['title']
save_this['subreddit'] = df['subreddit']
print("!!! FINALLY COMPLETED !!!")
save_this.to_csv('final_R_data.csv',index=False)