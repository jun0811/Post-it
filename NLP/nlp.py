import pandas as pd
from pyarrow import csv
from bs4 import BeautifulSoup
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk  # pip install nltk
import sys

sys.stdin = open('categoies.txt')

#  정해놓은 분류에 따른 카테고리 딕셔너리 만들기 
categoies = []
for _ in range(10):
    arr = list(input().split())
    categoies.append(arr)
# print(categoies)
i = 0
category_dict = {}
for category in categoies:   
    for item in category:
        category_dict[item] = i
    i+=1

########################################
# 결과값 담을 배열
result = []

# 태그들 불러오기.
df = pd.read_csv('tags.csv')
df = df['tags'].values.tolist()
dff = df
values = []


# 태그들 앞글자 대문자로 변경
for value in dff:
    values.append(value.capitalize())
    # values.append(value.upper())

# 딕셔너리로 만들기.
key_val = [df, values]
tag_dict = dict(zip(*key_val))

data = pd.read_csv("./sof.csv")
df = pd.DataFrame(data, columns=["id","title","body", "tags"])
# 새로운 title+body column 
df['t-body'] ='<p>' + df['title'] + '</p>' + df['body']

for row_index, row in df.iterrows(): # 10개만 일단
    categoies = [0] * 10   # 카테고리 초기화 
    soup = BeautifulSoup(row["t-body"],'html.parser')
    tags = BeautifulSoup(row["tags"],'html.parser')
    tags = str(tags).split("|")
    # p태그만 가져오기 
    a = str(soup.find_all("p"))
    # print(a)
    # code 태그 지우기
    while a.find('<code>') >= 0:
        a = a.replace(a[a.find('<code>'):a.find('</code>')+7], '')
# regular expression을 이용하여 태그 다 지우기.
    a = re.sub('<.+?>', '', a, 0).strip()
# 특수문자 제거 re.sub('패턴', 교체함수, '문자열', 바꿀횟수)
    a = re.sub('[,/\?:^$@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\…》]', '', a)
    tokens = nltk.word_tokenize(a)

    tmp = []
    for index, token in enumerate(tokens):
        # 카테고리 딕셔너리에 맞게 가중치를 +1씩 category list  
        if token.lower() in category_dict.keys():
            categoies[category_dict[token.lower()]] +=1 
        if tag_dict.get(token):
            tmp.append(token.lower()) 
            tokens.pop(index)
    # 해당 글에 태그당 카테고리 점수를 준다.
    for tag in tags:
        if tag in category_dict.keys():
            categoies[category_dict[tag]] += 1
    
    maxV = max(categoies)
    # nltk.download('punkt') 를 실행하여 Punket Tokenizer Models (13MB) 를 다운로드 해줍니다.
    # 품사 태깅을 하려면 먼저 nltk.download('averaged_perceptron_tagger') 로 태깅에 필요한 자원을 다운로드 해줍니다.
    tagged = nltk.pos_tag(tokens)
    for tag in tagged:
        if tag[1] == 'NN' or tag[1] == 'NNP':
            tmp.append(tag[0])
    if categoies.count(maxV)<=1:
        result.append([row["id"], tmp+tags, categoies.index(maxV)])

print(result)

# # [ id , [타이틀, 본문, 이글에 태그까지], 카테고리 ] 

# # 카테고리 => 
# max -> count -> 1 일떄만 데이터베이스에 집어넣고 나머지 버림 