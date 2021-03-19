#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score


# In[3]:


data_x = []
data_y = []
with open('../NLP/newlistfile.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx=0
    for row in reader:
        if(idx==40000):
            break
        words = row[1][2:len(row[1])-2].replace("\"","").replace("\\","").replace("'", "").split(", ")
        data_x.append(' '.join(words))
        data_y.append(row[2])
        idx+=1


# In[4]:


# DTM train_x 데이터
cv = CountVectorizer() # lowercase true default
transformed_data_x = cv.fit_transform(data_x)


# In[5]:


# TF-IDF
tv = TfidfVectorizer().fit(data_x)
tfidf_transformed_data_x = tv.transform(data_x)


# In[6]:


# train,test 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(transformed_data_x, data_y, test_size=0.2) # train,test로 나눔
tf_train_x, tf_test_x, tf_train_y, tf_test_y = train_test_split(tfidf_transformed_data_x, data_y, test_size=0.2)


# In[10]:


#scaler = StandardScaler() # 스케일링 아직 잘 모름..
#scaler.fit(train_x, train_y)

#X_train_transformed = scaler.transform(train_x)
#X_test_transformed = scaler.transform(train_y)

cv_mnb = nb.MultinomialNB().fit(train_x.todense(),train_y) # Multinomial Naive Bayes
tf_mnb = nb.MultinomialNB().fit(tf_train_x.todense(), tf_train_y)
"""
bnb = nb.BernoulliNB()
bnb.fit(train_x, train_y)

comnb = nb.ComplementNB()
comnb.fit(train_x, train_y)
"""


# In[ ]:


cv_gnb = nb.GaussianNB().fit(train_x.todense(), train_y)
tf_gnb = nb.GaussianNB().fit(tf_train_x.todense(), tf_train_y)


# In[ ]:


cv_cnb = nb.CategoricalNB().fit(train_x.todense(), train_y)
tf_cnb = nb.CategoricalNB().fit(tf_train_x.todense(), tf_train_y)


# In[13]:


cv_mnb_pred_y = cv_mnb.predict(test_x)
tf_mnb_pred_y = tf_mnb.predict(tf_test_x)

cv_acc = accuracy_score(test_y, cv_mnb_pred_y) # 테스트 모델과 실제 측정값 데이터 정확도 측정
tf_acc = accuracy_score(test_y, tf_mnb_pred_y)
print("cv_acc : {}, tf_acc : {}".format(cv_acc, tf_acc))


# In[ ]:


cv_gnb_pred_y = cv_gnb.predict(test_x)
tf_gnb_pred_y = tf_gnb.predict(tf_test_x)

cv_acc = accuracy_score(test_y, cv_mnb_pred_y) # 테스트 모델과 실제 측정값 데이터 정확도 측정
tf_acc = accuracy_score(test_y, tf_mnb_pred_y)
print("cv_acc : {}, tf_acc : {}".format(cv_acc, tf_acc))


# In[ ]:





# In[11]:


cv_mnb_pred_y = cv_mnb.predict(test_x)
tf_mnb_pred_y = tf_mnb.predict(tf_test_x)

print("========== CounterVectorizer =============")
print(confusion_matrix(test_y, cv_mnb_pred_y))
print(classification_report(test_y, cv_mnb_pred_y))
print("========== TF- IDF Vectorizer =============")
print(confusion_matrix(test_y, tf_mnb_pred_y))
print(classification_report(test_y, tf_mnb_pred_y))


# In[ ]:


cv_gnb_pred_y = cv_gnb.predict(test_x)
tf_gnb_pred_y = tf_gnb.predict(test_x)

print("========== CounterVectorizer =============")
print(confusion_matrix(test_y, cv_gnb_pred_y))
print(classification_report(test_y, cv_gnb_pred_y))
print("========== TF- IDF Vectorizer =============")
print(confusion_matrix(test_y, tf_gnb_pred_y))
print(classification_report(test_y, tf_gnb_pred_y))


# In[ ]:


cv_cnb_pred_y = cv_cnb.predict(test_x)
tf_cnb_pred_y = tf_cnb.predict(test_x)

print("========== CounterVectorizer =============")
print(confusion_matrix(test_y, cv_cnb_pred_y))
print(classification_report(test_y, cv_cnb_pred_y))
print("========== TF- IDF Vectorizer =============")
print(confusion_matrix(test_y, tf_cnb_pred_y))
print(classification_report(test_y, tf_cnb_pred_y))


# In[66]:


#bnb_pred_y = bnb.predict(test_x)

#print(confusion_matrix(test_y, bnb_pred_y))
#print(classification_report(test_y, bnb_pred_y))
#에러


# In[67]:


#comnb_pred_y = comnb.predict(test_x)

#print(confusion_matrix(test_y, comnb_pred_y))
#print(classification_report(test_y, comnb_pred_y))
#에러


# In[ ]:




