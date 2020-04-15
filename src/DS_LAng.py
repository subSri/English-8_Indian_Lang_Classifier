#!/usr/bin/env python
# coding: utf-8

# In[117]:


import logging
import pandas as pd
import numpy as np
from numpy import random

import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


# In[118]:


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


# In[119]:


from nltk.tokenize import word_tokenize
inputfile = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\inputLangData.txt','r',encoding="utf8")
text = inputfile.read()
input_list = text.split()
print(len(input_list))
inputfile.close()
# print(input_list)


# In[120]:


#input_testing

import bs4
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


# In[1]:


from nltk.tokenize import word_tokenize
inputfile = open(r'C:\Users\SUBHAM\Desktop\DS-LanguageClass\TestinputLangData.txt','r',encoding="utf8")
text = inputfile.read()
test_input_list = text.split()
print(len(test_input_list))
inputfile.close()
print(test_input_list[0:])


# In[122]:


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
label_list = text.split()
print(len(label_list))
labelfile.close()
# print(label_list)


# In[123]:


#Input test Labels

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
test_label_list = text.split()
print(len(test_label_list))
labelfile.close()
# print(test_label_list)


# In[ ]:





# In[124]:


map_input_label = {}
total_vocab=set()
new_input=[]
new_label=[]
X_trainActual=[]
y_trainActual=[]
for m in range(len(label_list)):
    if label_list[m] in preffered_labels:
        allstrings = []
        allstrings = [input_list[m][i: j] for i in range(len(input_list[m])) for j in range(i+3, len(input_list[m]) + 1)] 
        buf1 = []
        if len(allstrings)>0:
            for n in allstrings:
                if len(n)<=6:
                    new_input.append(n)
                    new_label.append(label_list[m])
                    buf1.append(n)
            X_trainActual.append(buf1)
            y_trainActual.append(label_list[m])
#         myfile = open("C:\\Users\\SUBHAM\\Desktop\\DS-LanguageClass\\"+label_list[i]+".txt",'a',encoding="utf8")
#         total_vocab.add(n)
#         if label_list[m] in map_input_label:
#             if n in map_input_label[label_list[m]]:
#                 map_input_label[label_list[m]][n] += 1
#             else:
#                 map_input_label[label_list[m]][n] = 1
#         else:
#             map_input_label[label_list[m]]={}
                
       
# total_vocab = list(total_vocab)
print(len(X_trainActual))
print(len(y_trainActual))
print(len(new_input),len(new_label))


# In[ ]:





# In[125]:


#X_test and Y_test

X_test=[]
y_test=[]
X_testActual=[]
y_testActual=[]
refresh_test_list = []
for m in range(len(test_label_list)):
    if test_label_list[m] in preffered_labels:
        allstrings = []
        allstrings = [test_input_list[m][i: j] for i in range(len(test_input_list[m])) for j in range(i+3, len(test_input_list[m]) + 1)] 
        buf = []
        if len(allstrings)>0:
            for n in allstrings:
                if len(n)<=6:
                    X_test.append(n)
                    y_test.append(test_label_list[m])
                    buf.append(n)
            refresh_test_list.append(test_input_list[m])
            X_testActual.append(buf)
            y_testActual.append(test_label_list[m])


# In[126]:


print(len(X_testActual),len(y_testActual))
# print(X_testActual)
print(X_test)
print()


# In[ ]:





# In[ ]:





# In[110]:


#0.5714285714285714

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
X_train = new_input
y_train = new_label
nb.fit(X_train, y_train)

#testing
from sklearn.metrics import classification_report

y_pred = nb.predict(X_test)


# In[111]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfTransformer

# nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', RandomForestClassifier(n_estimators=1000, random_state=0)),
#               ])
# X_train = new_input
# y_train = new_label
# nb.fit(X_train, y_train)

# #testing
# from sklearn.metrics import classification_report

# y_pred = nb.predict(X_test)


# In[127]:


# 0.5990062111801242

from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
#                 ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
X_train = new_input
y_train = new_label
sgd.fit(X_train, y_train)

# %%time

y_pred = sgd.predict(X_test)


# In[93]:


# #array classifier

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# X_train = X_trainActual
# y_train = y_trainActual
# clf.fit(X_train, y_train)

# y_pred=clf.predict(X_test)

# # clf_pf = GaussianNB()
# # clf_pf.partial_fit(X, Y, np.unique(Y))

# # print(clf_pf.predict([[-0.8, -1]]))


# In[128]:


from itertools import islice 
  
y_predActual = []
def convert(lst, var_lst): 
    idx = 0
    for var_len in var_lst: 
        yield np.array(lst[idx : idx + var_len])
        idx += var_len 
var_lst = [len(i) for i in X_testActual]
# print()
y_predActual = list(convert(np.array(y_pred),np.array(var_lst)))
y_predActual


# In[129]:


len(y_predActual)
print(len(y_testActual))


# In[130]:


y_predConfined=[]
for wordlist in y_predActual:
  
    countele = np.unique(wordlist, return_counts=True)
    
    index = list(countele[1]).index(max(countele[1]))
#     print("Countele",countele,"index",index)
    y_predConfined.append(countele[0][index])
#     y_predExtract.append(finalele)
    
len(y_predConfined)


# In[131]:


print('accuracy %s' % accuracy_score(y_predConfined, y_testActual))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




