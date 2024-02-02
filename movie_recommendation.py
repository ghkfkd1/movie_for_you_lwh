import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

def getRecommenation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[:11]
    movieIdx = [i[0] for i in simScore]
    recmovielist = df_reviews.iloc[movieIdx, 0] #영화제목 10개 받아서 리턴
    return recmovielist[1:11]

df_reviews = pd.read_csv('./cleaned_one_review.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)
ref_idx = 0
print(df_reviews.iloc[ref_idx, 0]) #스파이더맨
consine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
print(consine_sim[0])
print(len(consine_sim))
recommendation = getRecommenation(consine_sim)
print(recommendation)
