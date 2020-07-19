#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from nltk.corpus import stopwords 
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

#1)preprocessing
#read data / split string
#mood=0:negative,mood=1:positive
test = pd.DataFrame(columns = ['mood','sentence'])
testfile = open('testing_label.txt','r')
line = testfile.readline()
while line:
    if line != '\n':
        temp = line.split("#####", 1)
        new = pd.DataFrame({'mood':temp[0],'sentence':temp[1]},index=[1])
        test = test.append(new,ignore_index=True)
    line = testfile.readline()
testfile.close()

train = pd.DataFrame(columns = ['mood','sentence'])
trainfile = open('training_label.txt','r')
line = trainfile.readline()
i = 0
while line:
    if line != '\n':
        i+=1
        temp = line.split("+++$+++", 1)
        new = pd.DataFrame({'mood':temp[0],'sentence':temp[1]},index=[1])
        train = train.append(new,ignore_index=True)
    if i == 100000 : break 
    line = trainfile.readline()
trainfile.close()

#remove stop words
#stop_words = set(stopwords.words('english')) 
stop_words = [',','.','..','...',"'",'`',':','1','2','3','4','5','6','7','8','9','0',
              '00','000','0000','000pv','day','so','all','up','got','today','from','one',
              'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
              't','u','v','w','x','y','z','im','you','your','u','ur','they','an','him',
              'we','he','she','me','my','his','her','it','is','am','are','was','were',
              'btw','thing','be','have','has','had','do','does','did','ll','re','ve',
              'this','that','there','say','says','said','guys','to','for','in','at',
              'when','the','on','its','and','in','of','some','someone','before','after',
              'with','been','being','which','them','their','our','us','left','10','these',
              '30','site','online','12','da','room','20','sometimes','11','sat','15','google'
              '24','info','sit','web','website','09','17','18','33','50','session','16','21',
              '25','b4','pm','13','333','45','70','2009','month','yr','yrs','23',
              'txt','06','22','26','29','31','37','80','07','19','28',
              '32','34','35','36','38','48','52','69','79','89','95','97']
#stop_words = stop_words.union(new_stopwords)
for i in range(len(test)):
    test_sent = test['sentence'][i] 
    word_tokens = word_tokenize(test_sent) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    sentence='' 
    for w in word_tokens: 
        if w not in stop_words: 
            sentence += w + ' '
    test['sentence'][i] = sentence
for i in range(len(train)):
    train_sent = train['sentence'][i] 
    word_tokens = word_tokenize(train_sent) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]  
    sentence=''
    for w in word_tokens: 
        if w not in stop_words: 
            sentence += w + ' ' 
    train['sentence'][i] = sentence

train_y = train.mood
train_x = train.sentence
test_y = test.mood
test_x = test.sentence 

#word embeddings
tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
train_fit = tv.fit_transform(train_x)
test_fit = tv.transform(test_x)
#tv.get_feature_names()

print('with costom stop words and tuned parameter')
#2)build model
#AdaBoost
adac = AdaBoostClassifier(learning_rate=0.5,n_estimators=1000, random_state=0)
adac.fit(train_fit,train_y)
preda_y = pd.DataFrame(adac.predict(test_fit),columns = ['mood'])
preda_y['mood'] = preda_y['mood'].map(str.rstrip)
print('AdaBoost')
print(classification_report(test_y,preda_y))
#XGBoost
xgbc = XGBClassifier(learning_rate=0.5,n_estimators=1700,max_depth=9,min_child_weight=1,gamma=0,
                     subsample=0.8,colsample_bytree=0.8,nthread=4,seed=28)
xgbc.fit(train_fit,train_y)
predx_y = pd.DataFrame(xgbc.predict(test_fit),columns = ['mood'])
predx_y['mood'] = predx_y['mood'].map(str.rstrip)
print('XGBoost')
print(classification_report(test_y,predx_y))
