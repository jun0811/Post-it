import abc
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import nltk  # pip install nltk


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

f = open("./TestCase.txt", 'r')

# with open("./TestCase.txt") as fp:
#     soup = BeautifulSoup(fp, 'txt.parser')

soup = BeautifulSoup(f.read())

# p태그만 뽑아오기.
a = str(soup.find_all("p"))

# code 태그 지우기
while a.find('<code>') >= 0:
    a = a.replace(a[a.find('<code>'):a.find('</code>')+7], '')

# regular expression을 이용하여 태그 다 지우기.
a = re.sub('<.+?>', '', a, 0).strip()

# 특수문자 제거 re.sub('패턴', 교체함수, '문자열', 바꿀횟수)
a = re.sub('[,/\?:^$@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\…》]', '', a)
print(a)

# nltk.download('punkt') 를 실행하여 Punket Tokenizer Models (13MB) 를 다운로드 해줍니다.
# 토큰으로 자르기.
tokens = nltk.word_tokenize(a)

for index, token in enumerate(tokens):
    if tag_dict.get(token):
        result.append(tag_dict.get(token))
        tokens.pop(index)


# 품사 태깅을 하려면 먼저 nltk.download('averaged_perceptron_tagger') 로 태깅에 필요한 자원을 다운로드 해줍니다.
tagged = nltk.pos_tag(tokens)
print(tagged)

for tag in tagged:
    if tag[1] == 'NN' or tag[1] == 'NNP':
        result.append(tag[0])
        # print(tag[0])

print(result)
