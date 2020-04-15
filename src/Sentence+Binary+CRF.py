#!/usr/bin/env python
# coding: utf-8

# In[143]:


import logging
import pandas as pd
import numpy as np
from numpy import random
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
preffered_labels = ['en',  'kn', 'ta', 'gu', 'bn', 'te', 'hi', 'mr','ml']


# # TrainNgrams

# In[144]:



import pandas as pd
infile = open(r'C:\Users\SUBHAM\Downloads\IDRBT-FIRE-master\IDRBT-FIRE-master\dataFIRE\ngramsTrain.txt',encoding="utf8")
contents = infile.read()

docs = contents.split("\n")
# print(docs[0])
infile.close()


# In[145]:


X_train = []
y_train = []

for i in range(len(docs)-1):
    splittedDoc = docs[i].split(",")
#     print(splittedDoc[1])
    X_train.append(splittedDoc[1])#.split(" "))
    y_train.append(splittedDoc[0])
# print((X_train),(y_train))


# # TestNgrams

# In[146]:


#input_testing

import bs4
import re
from bs4 import BeautifulSoup
infile = open(r'C:\Users\SUBHAM\Downloads\IDRBT-FIRE-master\IDRBT-FIRE-master\dataFIRE\InputTesting.txt',encoding="utf8")
contents = infile.read()

input_file = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\TestinputLangData.txt','w',encoding="utf8")

soup = BeautifulSoup(contents)
titles = soup.find_all('utterance')
for title in titles:
    input_file.write(title.get_text())

input_file.close()
infile.close()

from nltk.tokenize import word_tokenize
inputfile = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\TestinputLangData.txt','r',encoding="utf8")
text = inputfile.read()

# print(text)


# In[147]:


test_input_list = re.split('\t|\n',text)

inputfile.close()
my_X_train = []
for i in test_input_list:
    if i!='':
        my_X_train.append(i)
    
# print(len(my_X_train))
# print(my_X_train)
XX_train = []
for i in my_X_train:
    XX_train.append(i.split())

# print(XX_train)


# In[148]:


import re
import bs4
from bs4 import BeautifulSoup
infile = open(r'C:\Users\SUBHAM\Downloads\IDRBT-FIRE-master\IDRBT-FIRE-master\dataFIRE\AnnotationTesting.txt',encoding="utf8")
contents = infile.read()

input_file = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\TestmyDSInputLabel.txt','w',encoding="utf8")

soup = BeautifulSoup(contents)
titles = soup.find_all('utterance')
for title in titles:
    input_file.write(title.get_text())

input_file.close()

labelfile = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\TestmyDSInputLabel.txt','r',encoding="utf8")
text = labelfile.read()
test_label_list = re.split('\t|\n',text)

labelfile.close()

my_Y_train = []
for i in test_label_list:
    if i!='':
        my_Y_train.append(i)

print(len(my_Y_train))


YY_train = []
for i in my_Y_train:
    YY_train.append(i.split())

# print(YY_train)


# In[149]:


X_test = []
y_test = []
Xx_test = []
Xraw_test = []
for i in range(len(YY_train)):
    x=""
    xx=[]
    for j in range(len(YY_train[i])):
        if (XX_train[i][j]).isalpha()==True and YY_train[i][j] in preffered_labels:
#             print(XX_train[i][j])
            Xraw_test.append(XX_train[i][j])
            xx.append(XX_train[i][j])
            x+=XX_train[i][j]+" "
            y_test.append(YY_train[i][j])
    X_test.append(x)
    Xx_test.append(xx)
            

print(len(X_test),len(y_test))


# In[150]:


def gen_ngrams(data):
    allstrings = ""
    for i in range(len(data)):
        for j in range(i+1,len(data)+1):
            if len(data[i:j])<=5:
                allstrings =allstrings+" "+str(data[i:j])
    return allstrings


# In[151]:


x_test_ngrams = []
for i in Xx_test:
    sent_ngrams = ""
    for j in i:
        sent_ngrams+= gen_ngrams(j)
    x_test_ngrams.append(sent_ngrams)
    
print(len(x_test_ngrams))


# In[152]:


import pandas as pd
intestfile = open(r'C:\Users\SUBHAM\Downloads\IDRBT-FIRE-master\IDRBT-FIRE-master\dataFIRE\ngramsTest.txt',encoding="utf8")
testcontents = intestfile.read()

testdocs = testcontents.split("\n")
# print(docs[0])
intestfile.close()


# In[ ]:





# In[153]:


Xng_test = []
yng_test = []

for i in range(len(testdocs)-1):
    splittedDoc = testdocs[i].split(",")
#     print(splittedDoc[1])
    Xng_test.append(splittedDoc[1])#.split(" "))
    yng_test.append(splittedDoc[0])
print(len(Xng_test),len(yng_test))


# In[154]:


## Naive Bayes 0.7102564102564103
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])


# In[155]:


nb.fit(X_train, y_train)
y_pred_ng = nb.predict(Xng_test)
# y_pred_ng


# In[156]:


print('accuracy %s' % accuracy_score(y_pred_ng, yng_test))
# print(classification_report(y_test, y_pred,target_names=my_tags))


# In[157]:


nb.fit(X_train, y_train)
y_pred = nb.predict(x_test_ngrams)
# y_pred


# In[158]:


# 0.6935897435897436

# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfTransformer

# nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', LogisticRegression(random_state=100)),
#               ])
# nb.fit(X_train, y_train)

# from sklearn.metrics import classification_report
# y_pred = nb.predict(X_test)


# In[159]:


# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfTransformer

# #Importing MLPClassifier
# from sklearn.neural_network import MLPClassifier

# #Initializing the MLPClassifier
# classifier = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf',  MLPClassifier(hidden_layer_sizes=(100), max_iter=300,activation = 'relu',solver='adam',random_state=1)),
#               ])


# In[ ]:





# In[160]:


# from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfTransformer


# classifier = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf',  GaussianNB()),
#               ])
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# In[ ]:





# In[161]:



def binClassifier(Xx_test,y_pred):
    new_X_test=[]
    new_y_test = []
    pred_label=[]

#     language_list=['hi','gu','kn','ml','mr','bn','ta','te']

    for y in range(len(y_pred)):
        word_pred=[]
        if y_pred[y]!='en_':
#             if  y_pred[y] in language_list:
            filename= y_pred[y]+'.sav'
            model=pickle.load(open(filename,'rb'))
            
                
            
            for v in Xx_test[y]:
                doc_ngrams=""
                doc_ngrams=gen_ngrams(v)
          
                
                new_X_test.append(doc_ngrams)
                pred_label.append(list(model.predict([doc_ngrams]))[0])
        else:
            
            for v in Xx_test[y]:
                doc_ngrams=""
                doc_ngrams=gen_ngrams(v)
          
                
                new_X_test.append(doc_ngrams)
                pred_label.append('en')
#         new_y_test.append(word_lab_map[v])
            

    return [new_X_test,pred_label]


# In[162]:


# def getWordsfromNgrams(p):
#     #print(len(p))
#     word_list=[]
#     word=''
#     for gram in range(len(p)-1):
#         if len(p[gram])==1 and p[gram+1] and len(p[gram+1])>1:
#             word+=p[gram]
#             word_list.append(word)
#             word=''
#         elif len(p[gram])==1 and len(p[gram+1])==1:
#             word+=p[gram]
#         else:
#             pass
#     return word_list


# In[ ]:





# In[163]:


new_X_test,pred_label = binClassifier(Xx_test,y_pred)


# In[ ]:





# In[164]:


print('accuracy of bin classifier %s' % accuracy_score(pred_label, y_test))


# # CRF

# In[165]:


#Training CRF
#input_training
preffered_labels = ['en',  'kn', 'ta', 'gu', 'bn', 'te', 'hi', 'mr','ml']
import bs4
from bs4 import BeautifulSoup
infile = open(r'C:\Users\SUBHAM\Downloads\IDRBT-FIRE-master\IDRBT-FIRE-master\dataFIRE\InputTraining.txt',encoding="utf8")
contents = infile.read()

input_file = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\inputLangData.txt','w',encoding="utf8")

soup = BeautifulSoup(contents)
titles = soup.find_all('utterance')
for title in titles:
    input_file.write(title.get_text())

input_file.close()
infile.close()


from nltk.tokenize import word_tokenize
inputfile = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\inputLangData.txt','r',encoding="utf8")
text = inputfile.read()


input_list = re.split('\t|\n',text)

inputfile.close()
my_X_train = []
for i in input_list:
    if i!='':
        my_X_train.append(i)
    
# print(len(my_X_train))
# print(my_X_train)
XX_train = []
for i in my_X_train:
    XX_train.append(i.split())
    


# In[166]:


#Input Labels

import bs4
from bs4 import BeautifulSoup
infile = open(r'C:\Users\SUBHAM\Downloads\IDRBT-FIRE-master\IDRBT-FIRE-master\dataFIRE\AnnotationTraining.txt',encoding="utf8")
contents = infile.read()

input_file = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\myDSInput.txt','w',encoding="utf8")

soup = BeautifulSoup(contents)
titles = soup.find_all('utterance')
for title in titles:
    input_file.write(title.get_text())

input_file.close()

labelfile = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\myDSInput.txt','r',encoding="utf8")
text = labelfile.read()
label_list = re.split('\t|\n',text)

labelfile.close()

my_Y_train = []
for i in label_list:
    if i!='':
        my_Y_train.append(i)




YY_train = []
for i in my_Y_train:
    YY_train.append(i.split())
print(len(YY_train))
labelfile.close()


# In[167]:


eX_train = []
way_train = []
Xx_train = []
Xraw_train = []
for i in range(len(YY_train)):
    x=""
    xx=[]
    for j in range(len(YY_train[i])):
        if (XX_train[i][j]).isalpha()==True and YY_train[i][j] in preffered_labels:
#             print(XX_train[i][j])
            Xraw_train.append(XX_train[i][j])
            xx.append(XX_train[i][j])
            x+=XX_train[i][j]+" "
            way_train.append(YY_train[i][j])
    eX_train.append(x)
    Xx_train.append(xx)
            


# In[168]:


x_train_ngrams = []
for i in Xx_train:
    sent_ngrams = ""
    for j in i:
        sent_ngrams+= gen_ngrams(j)
    x_train_ngrams.append(sent_ngrams)
    
print(len(x_train_ngrams))


# In[169]:


nb.fit(X_train, y_train)
y_pred_train = nb.predict(x_train_ngrams)
print(y_pred_train)


# In[ ]:





# In[170]:


new_X_train,train_pred_label = binClassifier(Xx_train,y_pred_train)


# In[171]:


import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# In[28]:


# crf_pred = [[i] for i in pred_label]
# crf_pred

# crf_data = [[i] for i in Xraw_test]
# crf_data

# crf_ytest = [[i] for i in y_test]


# In[113]:





crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    all_possible_states=True
)


# In[132]:


train_crf_pred = [[i] for i in train_pred_label]
crf_pred

train_crf_data = [[i] for i in Xraw_train]
crf_data

ground_crf_ytrain = [[i] for i in way_train]


# print(train_crf_data)


# In[133]:



crf.fit(train_crf_pred, ground_crf_ytrain)


# In[ ]:





# In[134]:


# crf.fit([["Hey","Hello"],["Hello","namste"],["namste","aapko"]],[["en","en"],["hi","en"],["mr","kl"]])


# In[135]:


# crf.fit([["Hey"],["Hello"],["namste"]],[["en"],["hi"],["mr"]]


# In[136]:


# crf.predict([[["Hello","bhaui"]],["aapko"],["namskar"]])


# In[141]:


crd_labels_predicted = crf.predict(crf_pred)
# print(crd_labels_predicted)


# In[142]:


print('CRF accuracy  %s' % accuracy_score(crd_labels_predicted,crf_ytest))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[424]:


# k=[]
# for y in range(len(y_pred)):
#     q=y_pred[y]
#     if y_pred[y]=='en_':
#         k.append('en')
#     else:

#         lang=y_pred[y]+'.sav'
#         loaded_model=pickle.load(open(lang,'rb'))
#         for m in Xx_test[y]:
#             for j in m:
#                 doc_ngrams=gen_ngrams(j)
#                 k.append(list(loaded_model.predict(doc_ngrams))[0])
# from sklearn import metrics
# #print(metrics.accuracy_score())
# print(len(y_test),len(k))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




