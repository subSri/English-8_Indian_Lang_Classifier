#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:





# In[81]:


f = open(r"C:\Users\SUBHAM\Desktop\DS-LanguageClass\eng2.txt")
cont = f.read()

eng_data = cont.split("\n")

f.close()
eng_data = eng_data[0:len(eng_data)-1]
# print(eng_data)
eng_label = ['en']*len(eng_data)


# In[124]:


f = open(r"C:\Users\SUBHAM\Desktop\DS-LanguageClass\maratiW.txt",encoding="utf8")
cont = f.read()

beng_data = cont.split("\n")
beng_data = beng_data[0:len(beng_data)-2]
f.close()
# print(beng_data)


# In[125]:


beng_ngram_data = []

for m in range(len(beng_data)):
    i=0
    j=0
    allstrings = []
    allstrings = [beng_data[m][i:j] for i in range(len(beng_data[m])) for j in range(i+1,len(beng_data[m])+1)] 
#     for i in range(len(beng_data[m])):
#         for j in range(len(beng_data[m])):
#             if i+j<=len(beng_data[m]):
#                 allstrings.append(beng_data[m][j:i+j])
                          
    buf1 = ""
    if len(allstrings)>0:
        for n in allstrings:
            if len(n)<=5:
                buf1=buf1+" "+n
        beng_ngram_data.append(buf1)

print(beng_ngram_data)
beng_label = ['mr']*len(beng_ngram_data)


# In[127]:


X = beng_ngram_data + eng_data
# print(X)
y = beng_label + eng_label
print(len(X),len(y))


# In[ ]:





# In[128]:


idx = np.random.permutation(len(X))
X_data = []
y_data = []
for i in idx:
    
    X_data.append(X[i-1])
    y_data.append(y[i-1])


# In[129]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)


# In[130]:


#0.0.9154334038054969 Bengali
#
nb= Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression(random_state=100)),
              ])


# In[131]:


#  #.0.9043340380549683 Bengali

# nb = Pipeline([('vect', CountVectorizer()),
# #                 ('tfidf', TfidfTransformer()),
#                 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
#                ])


# In[132]:




# #Initializing the MLPClassifier
# nb = Pipeline([('vect', CountVectorizer()),
# #                ('tfidf', TfidfTransformer()),
#                ('clf',  MLPClassifier(alpha = 0.7, max_iter=50)),
#               ])


# In[133]:


# #0.904334038054968385 Bengali

# b = Pipeline([('vect', CountVectorizer()),
# #                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#               ])


# In[134]:


nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)


# In[135]:


acc = accuracy_score(y_pred, y_test)
print('accuracy %s' % acc)

f1sc = f1_score(y_test, y_pred, average='micro')
print('f1 Score ',f1sc)


# In[136]:


import pickle
filename = 'mr_en_.sav'
pickle.dump(nb, open(filename, 'wb'))


# In[ ]:





# In[115]:


measures['Tamil']['Accuracy'] = acc
measures['Tamil']['F1-Score'] =  f1sc


# In[116]:


display(measures) #Logistic


# In[15]:



#measures = pd.DataFrame(np.zeros([2,9]),index=['Accuracy', 'F1-Score'],columns = ['Beng','Hindi','Eng','Kannada','Telugu','Tamil','Guj','Marathi','Malyalam'])


# In[ ]:




