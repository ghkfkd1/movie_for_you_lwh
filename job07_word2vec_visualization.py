import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus'] = False
rc('font', family= font_name)

embedding_model = Word2Vec.load('./models/word2vec_movie_review.model') #모델을 불러서 모델한테 키워드 줌
key_word = '사랑'
sim_word = embedding_model.wv.most_similar(key_word, topn=10) # 키워드 사랑과 가장 가까이 있는 단어 10개
print(sim_word)

#차원을 축소해서 그리기
vectors = []
labels = []

for label, _ in sim_word: #('이별', 0.6098082065582275) 에서 뒤에 숫자는 안씀
    labels.append(label) # 단어 저장
    vectors.append(embedding_model.wv[label]) # 단어에 부합하는 숫자를 저장

print(vectors[0]) # 좌표 100개 가지고 있음
print(len(vectors[0])) # 100: 차원 수가 100개이기 때문







df_vectors = pd.DataFrame(vectors)
print(df_vectors.head())

tsne_model = TSNE(perplexity=9, n_components=2, init='pca', n_iter=2500) #n_components:2차원으로 축소.
new_value = tsne_model.fit_transform(df_vectors) # 2차원 좌표를 만들어줌.
df_xy = pd.DataFrame({'words': labels, 'x':new_value[:, 0], 'y':new_value[:, 1]})

df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)
print(df_xy)
print(df_xy.shape)

plt.figure(figsize=(8, 8))
plt.scatter(0, 0, s=1500, marker='*') #s:사이즈 0,0 좌표에 별 그림

for i in range(len(df_xy)):
    a = df_xy.loc[[i, 10]]
    plt.plot(a.x, a.y, '-D', linewidth=1)
    plt.annotate(df_xy.words[i], xytext=(1, 1), xy=(df_xy.x[i], df_xy.y[i]), textcoords ='offset points', ha ='right', va= 'bottom')

plt.show()
