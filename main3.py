# 학습용 데이터
from sklearn import datasets
# 데이터를 학습용과 테스트용으로 나눌수 있는 함수
from sklearn.model_selection import train_test_split
# 데이터 표준화
from sklearn.preprocessing import StandardScaler
# Perceptron 머신러닝을 위한 클래스
from sklearn.linear_model import Perceptron
# 정확도 계산을 위한 함수
from sklearn.metrics import accuracy_score

# 파일 저장을 위해서..
import pickle
import numpy as np

from plotdregion import *

names = None

def step1_get_data():
    # 아이리스 데이터 추출
    iris = datasets.load_iris()
    # print(iris)
    # 꽃 정보 데이터 추출
    x = iris.data[:100, [2,3]] # 꽃잎정보
    y = iris.target[:100]      # 꽃 종류
    names = iris.target_names[:2] # 꽃 이름
    print(x[0])
    print(y[0])
    print(names[0])
    return x, y

def step2_learning():
    x, y = step1_get_data()
    # 학습 데이터와 테스트 데이터로 나눈다.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # 표준화작업 : 데이터들을 표준 정규분포로 변환하여
    # 적은 학습횟수와 높은 학습 정확도를 갖기 위해 하는 작업
    sc = StandardScaler()
    # 데이터를 표준화한다.
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    # 학습한다.
    ml = Perceptron(eta0 = 0.01, max_iter=40, random_state=0)
    ml.fit(x_train_std, y_train)
    # 학습 정확도를 확인해본다.
    x_test_std = sc.transform(x_test)
    y_pred = ml.predict(x_test_std)
    print("학습 정확도 : ", accuracy_score(y_test, y_pred))
    # 학습이 완료된 객체를 지정한다.
    with open('./4.Scikit-Perceptron/ml.dat','wb') as fp:
        pickle.dump(sc, fp)
        pickle.dump(ml, fp)

    print("학습 완료")

    #시각화를 위한 작업
    # X_combined_std = np.vstack((x_train_std, x_test_std))
    # y_combined_std = np.hstack((x_train, x_test))
    # plot_decision_region(x=x=_combined_std, y=y_combined_std, classifier = ml, text_idx = ramge(70, 100),title='perceptron)

def step3_using():
        # 학습이 완료된 객체를 복원한다.
    with open('./4.Scikit-Perceptron/ml.dat','rb') as fp:
        sc =pickle.load(fp)
        ml =pickle.load(fp)

    while True:
        a1 = input("꽃잎의 너비를 입력해주세요 :")
        a2 = input("꽃잎의 길이를 입력해주세요 :")

        x= np.array([[float(a1), float(a2)]])
        # 계산된 결과를 가져온다.
        x_std =sc.transform(x)
        y=ml.predict(x_std)
        if y[0] == 0:
            print("결과 : Iris-setosa")
        else:
            print("결과 : Iris-versicolor")  


if __name__=="__main__":
    step1_get_data()
    step2_learning()
    step3_using()