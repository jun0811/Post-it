#!/usr/bin/env python
# coding: utf-8

# In[63]:


import sklearn.neighbors as nb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[64]:


data_x = []
data_y = []
with open('../NLP/newlistfile.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        words = row[1][2:len(row[1])-2].replace("\"","").replace("\\","").replace("'", "").split(", ")
        data_x.append(' '.join(words))
        data_y.append(row[2])


# In[85]:


# DTM train_x 데이터
cv = CountVectorizer() # lowercase true default
transformed_data_x = cv.fit_transform(data_x)


# In[66]:


# TF-IDF
tv = TfidfVectorizer().fit(data_x)
tfidf_transformed_data_x = tv.transform(data_x)


# In[67]:


# train,test 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(transformed_data_x, data_y, test_size=0.2) # train,test로 나눔
tf_train_x, tf_test_x, tf_train_y, tf_test_y = train_test_split(tfidf_transformed_data_x, data_y, test_size=0.2)


# In[68]:


# 러닝
cv_ml = nb.KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski').fit(train_x, train_y)
tf_ml = nb.KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski').fit(tf_train_x, tf_train_y)
#n_neighbors 낮아지면 결정경계 복잡해짐, 높아지면 단순해짐


# In[73]:


# 테스트 모델과 실제 측정값 데이터 정확도 측정
cv_pred_y = cv_ml.predict(test_x)
tf_pred_y = tf_ml.predict(tf_test_x)
cv_acc = accuracy_score(test_y, cv_pred_y)
tf_acc = accuracy_score(test_y, tf_pred_y)
print("cv_acc : {} , tf_acc : {}".format(cv_acc, tf_acc))


# In[74]:


print(classification_report(test_y, cv_pred_y)) # CounterVectorizer


# In[75]:


print(classification_report(test_y, tf_pred_y)) # TFIDF


# In[59]:


import seaborn as sn # heatmap - Accuracy Score
import matplotlib.pyplot as plt

con_mat = confusion_matrix(y_true=test_y, y_pred=pred_y)

plt.figure(figsize=(10,10))
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square=True)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score : {0}'.format(acc)
plt.title(all_sample_title, size=18)
plt.show()


# In[61]:


train_acc = []
test_acc = []

for n in range(1,20): # 1~19 neighbors 성능체크
    clf = nb.KNeighborsClassifier(n_jobs=-1, n_neighbors=n)
    clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)
    train_acc.append(clf.score(train_x, train_y))
    test_acc.append((prediction==test_y).mean())


# In[62]:


plt.figure(figsize=(12, 9))
plt.plot(range(1, 20), train_acc, label='TRAIN set')
plt.plot(range(1, 20), test_acc, label='TEST set')
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, 20, step=1))
plt.legend()


# In[77]:


new_data_x = []
new_data_y = []
with open('../NLP/train_data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    idx=0
    for row in reader:
        if(idx==10000):
            break
        words = row[1][2:len(row[1])-2].replace("\"","").replace("\\","").replace("'", "").split(", ")
        new_data_x.append(' '.join(words))
        new_data_y.append(row[2])
        idx+=1


# In[87]:


# DTM 새로운 데이터 예측
new_transformed_data_x = cv.transform(new_data_x) # 새로운 데이터 transform


# In[91]:


# TF-IDF 새로운 데이터 예측
new_tfidf_transformed_data_x = tv.transform(new_data_x) # 새로운 데이터 transform


# In[92]:


# 테스트 모델과 실제 측정값 데이터 정확도 측정
new_cv_pred_y = cv_ml.predict(new_transformed_data_x)
new_tf_pred_y = tf_ml.predict(new_tfidf_transformed_data_x)
cv_acc = accuracy_score(new_data_y, new_cv_pred_y)
tf_acc = accuracy_score(new_data_y, new_tf_pred_y)
print("cv_acc : {} , tf_acc : {}".format(cv_acc, tf_acc))


# In[ ]:




