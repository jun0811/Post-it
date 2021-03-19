#!/usr/bin/env python
# coding: utf-8

# In[187]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[188]:


data_x = []
data_y = []
with open('../NLP/newlistfile.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        words = row[1][2:len(row[1])-2].replace("\"","").replace("\\","").replace("'", "").split(", ")
        data_x.append(' '.join(words))
        data_y.append(row[2])


# In[189]:


# DTM train_x 데이터
cv = CountVectorizer() # lowercase true default
transformed_data_x = cv.fit_transform(data_x)


# In[190]:


# TF-IDF
tv = TfidfVectorizer().fit(data_x)
tfidf_transformed_data_x = tv.transform(data_x)


# In[191]:


# train,test 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(transformed_data_x, data_y, test_size=0.2) # train,test로 나눔
tf_train_x, tf_test_x, tf_train_y, tf_test_y = train_test_split(tfidf_transformed_data_x, data_y, test_size=0.2)


# In[183]:


# fit model
# 다항 로지스틱 회귀 , multi_class = multinomial 고정
# C : Train Strenth -> 클수록 규제(regularization)가 약하대요
from sklearn.linear_model import LogisticRegression

cv_model = LogisticRegression(multi_class='multinomial',C=100000,solver='newton-cg').fit(train_x, train_y)
tf_model = LogisticRegression(multi_class='multinomial',C=100000,solver='newton-cg').fit(tf_train_x, tf_train_y)


# In[205]:


# 테스트셋 예측
cv_predicted = cv_model.predict(test_x)
tf_predicted = tf_model.predict(tf_test_x)


# In[207]:


# 정확도 추출
cv_acc = accuracy_score(test_y, cv_predicted) # 테스트 모델과 실제 측정값 데이터 정확도 측정
tf_acc = accuracy_score(test_y, tf_predicted)
print("cv_acc : {}, tf_acc : {}".format(cv_acc, tf_acc))


# In[172]:


import seaborn as sn # heatmap - Accuracy Score
import matplotlib.pyplot as plt

con_mat = confusion_matrix(y_true=test_y, y_pred=predicted)

plt.figure(figsize=(10,10))
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square=True)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score : {0}'.format(acc)
plt.title(all_sample_title, size=18)
plt.show()


# In[199]:


print(classification_report(test_y, cv_predicted)) # CounterVectorizer 성능


# In[200]:


print(classification_report(test_y, tf_predicted)) # TF-IDF 성능


# In[208]:


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


# In[209]:


# DTM 새로운 데이터 예측
new_transformed_data_x = cv.transform(new_data_x) # 새로운 데이터 transform


# In[210]:


# TF-IDF 새로운 데이터 예측
new_tfidf_transformed_data_x = tv.transform(new_data_x) # 새로운 데이터 transform


# In[211]:


# 테스트 모델과 실제 측정값 데이터 정확도 측정
new_cv_pred_y = cv_model.predict(new_transformed_data_x)
new_tf_pred_y = tf_model.predict(new_tfidf_transformed_data_x)
cv_acc = accuracy_score(new_data_y, new_cv_pred_y)
tf_acc = accuracy_score(new_data_y, new_tf_pred_y)
print("cv_acc : {} , tf_acc : {}".format(cv_acc, tf_acc))


# In[ ]:




