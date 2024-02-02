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

#영화 인덱스를 이용한 추천
#ref_idx = 0
#print(df_reviews.iloc[ref_idx, 0]) #스파이더맨
#consine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
#print(consine_sim[0])
#print(len(consine_sim))
#recommendation = getRecommenation(consine_sim)
#print(recommendation)



#keyword를 이용한 추천
embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
keyword = '마블'
sim_word = embedding_model.wv.most_similar(keyword, topn=10) #유사단어 10 개 받음
words = [keyword] # 마블 하나 들어있음
for word, _ in sim_word: # 유사 단어 10개 꺼내서 가장 유사한거 부터 추가
    words.append(word) # 단어 총 키워드까지 11개 포함
setence = []
count = 10
for word in words: # 11개 단어 를 setence에 10개 추가.
    setence = setence + [word] * count # count만큼 곱해서 더함: frequency 만들어줌
    count -= 1
setence = ' '.join(setence)
print(setence)
setence_vec = Tfidf.transform([setence])
cosine_sim = linear_kernel(setence_vec, Tfidf_matrix) # setence 벡터값과 모든 벡터값의 코사인 유사도 찾음
recommendation = getRecommenation(cosine_sim) # 10개 영화 인덱스를 추림

print(recommendation)