import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

form_window = uic.loadUiType('./movie_recommendation.ui')[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
        with open('./models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
        self.df_reviews = pd.read_csv('./cleaned_one_review.csv')
        self.titles = list(self.df_reviews['titles']) #콤보박스에 추가 하기 위해 리스트 만듬
        self.titles.sort()
        for title in self.titles: #콤보박스에 타이틀 추가
            self.comboBox.addItem(title)

        #자동완성 기능 추가
        model = QStringListModel()
        model.setStringList(self.titles)
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)
        ######################################

        self.comboBox.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommendation.clicked.connect(self.btn_slot) #버튼 누르면 버튼 슬롯에 연결됨

    def btn_slot(self):
        key_word = self.le_keyword.text() #라인 에디터에서 텍스트 읽어서 저장
        if key_word in self.titles:
            recommendation = self.recommendation_by_movie_title(key_word)
        else:
            recommendation = self.recommendation_by_keyword(key_word) # 키워드를 함수에 넘겨줌.
        if recommendation:
            self.lbl_recommendation.setText(recommendation)

    def combobox_slot(self):
        title = self.comboBox.currentText() #현재 출력된 텍스트 읽음
        recommendation = self.recommendation_by_movie_title(title)
        self.lbl_recommendation.setText(recommendation)

    def recommendation_by_keyword(self, key_word):
        try:
            sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)  # 유사단어 10 개 받음
        except: #찾고자 하는 단어가 없을 때.
            self.lbl_recommendation.setText('제가 모르는 단어에요. ㅠㅠ')
            return 0

        words = [key_word]  # 마블 하나 들어있음
        for word, _ in sim_word:  # 유사 단어 10개 꺼내서 가장 유사한거 부터 추가
            words.append(word)  # 단어 총 키워드까지 11개 포함
        setence = []
        count = 10
        for word in words:  # 11개 단어 를 setence에 10개 추가.
            setence = setence + [word] * count  # count만큼 곱해서 더함: frequency 만들어줌
            count -= 1
        setence = ' '.join(setence)
        print(setence)
        setence_vec = self.Tfidf.transform([setence])
        cosine_sim = linear_kernel(setence_vec, self.Tfidf_matrix)  # setence 벡터값과 모든 벡터값의 코사인 유사도 찾음
        recommendation = self.getRecommenation(cosine_sim)  # 10개 영화 인덱스를 추림
        recommendation = '\n'.join(list(recommendation)) #줄바꿈으로 이어붙임
        return recommendation

    def recommendation_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation = self.getRecommenation(cosine_sim)
        recommendation = '\n'.join(list(recommendation)) #줄바꿈으로 이어붙임
        return recommendation

    def getRecommenation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        movieIdx = [i[0] for i in simScore]
        recmovielist = self.df_reviews.iloc[movieIdx, 0]  # 영화제목 10개 받아서 리턴
        return recmovielist[1:11]


if __name__ == '__main__':  # main이면 실행 import 되면 실행 안됨.
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())  # 윈도우가 종료되면 종료됨.