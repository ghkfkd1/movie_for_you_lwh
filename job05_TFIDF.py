#TF : text frequency: 문장 안에서 단어 몇개 있는지
#DF : document frequency :문서 전체에서 단어 몇 개 있는지
# 모든 문장에 단어가 많이 나오면 오히려 유사도 깎임.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

df_reviews = pd.read_csv('./cleaned_one_review.csv')
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_reviews['reviews'])
print(Tfidf_matrix.shape)

with open('./models/tfidf.pickle','wb') as f:
    pickle.dump(Tfidf, f)

mmwrite('./models/Tfidf_movie_review.mtx',Tfidf_matrix)

#같은 방향에 있으면 cos 값이 같다.
#