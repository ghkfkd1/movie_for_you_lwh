#단어를 의미공간상 좌표를 부여해서 벡터화함.
#비슷한 의미의 단어가 비슷한 위치에 배치.

import pandas as pd
from gensim.models import Word2Vec

df_review = pd.read_csv('./cleaned_one_review.csv')
df_review.info()

reviews = list(df_review['reviews'])
print(reviews[0])

tokens = [] #형태소대로 자름
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])

embedding_model = Word2Vec(tokens, vector_size= 100, window=4, min_count=20, workers=4, epochs=100, sg=1)
#vector size: 100차원으로 줄임. 원래 몇차원?:7400 window:형태소 4개씩 봄. min_count:20개 이하 등장하면 학습 안함.
#workers : 드라이브 수 ???
#sg: 학습할 때 쓰는 알고리즘 종류
embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key)) #7440. AI는 7440개의 단어만 알고 있음.

