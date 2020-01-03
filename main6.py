
# def step2_learning():
#     x, y = step1_get_data()
#     # 학습 데이터와 테스트 데이터로 나눈다.
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#     # 표준화작업 : 데이터들을 표준 정규분포로 변환하여
#     # 적은 학습횟수와 높은 학습 정확도를 갖기 위해 하는 작업
#     sc = StandardScaler()
#     # 데이터를 표준화한다.
#     sc.fit(x_train)
#     x_train_std = sc.transform(x_train)
#     # 학습한다.
#     # criterion : 불순도 측정 방식, entropy, gini
#     # max_depth: 노드 깊이의 최대 값
#     ml = DecisionTreeClassifier(criterion='entropy', max_depth= 3 , random_state=0)
#     ml.fit(x_train_std, y_train)
#     # 학습 정확도를 확인해본다.
#     x_test_std = sc.transform(x_test)
#     y_pred = ml.predict(x_test_std)
#     print("학습 정확도 : ", accuracy_score(y_test, y_pred))
#     # 학습이 완료된 객체를 지정한다.
#     with open('./7.Scikit-Tree/ml.dat','wb') as fp:
#         pickle.dump(sc, fp)
#         pickle.dump(ml, fp)

    # print("학습 완료")

#학습용 데이터
from sklearn import datasets
#데이터를 학습용, 테스트용으로 나누는 함수
from sklearn.model_selection import train_test_split
# 데이터 표준화
from sklearn.preprocessing import StandardScaler
#perceptron 머신러닝위한 클라스
from sklearn.linear_model import Perceptron 
#로지스트 회귀를 위한 클라스
from sklearn.linear_model import LogisticRegression
#SVM을 위한 클래스 
from sklearn.svm import SVC
#의사결정나무를 위한 클래스
from sklearn.tree import DecisionTreeClassifier
#그리드서치 클래스 
from sklearn.model_selection import GridSearchCV
#교차검증 
#from sklearn.model_selection import KFold
#정확도 계산을 위한 함수
from sklearn.metrics import accuracy_score
#파일저장을 위한
import pickle
import numpy as np  

from plotdregion import *

names = None

def step1_get_data():
    #아이리스데이터추출
    iris = datasets.load_iris()
    #print(iris)

    X= iris.data[:150,[2,3]] #꽃잎정보
    y= iris.target[:150] #꽃정보
    names = iris.target_names[:3] #꽃이름

    return X,y


def step2_learning() :
    X,y = step1_get_data()
    #학습데이터와 테스트데이터로 나눈다
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0 )
    #표준화 작업 : 데이터들을 표준 정규분포로 변환하여 적은 학습 횟수와 높은 학습정확도를 갖기 위해 하는 작업
    sc = StandardScaler()
    #데이터를 표준화 한다
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    #학습한다.
    #ml = Perceptron(eta0=0.01, max_iter=40, random_state=0)
    # ml = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    ml = DecisionTreeClassifier()
    parameters = {'max_depth':[1,2,3,4,5,6,7], 'min_samples_split':[2,3]}
    grid_ml = GridSearchCV(ml, param_grid=parameters, cv=3, refit=True)
    grid_ml.fit(X_train_std, y_train)


    #학습 정확도를 확인해본다. 
    print('GridSearchCV 최적 파라미터:', grid_ml.best_params_)
    print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_ml.best_score_))
    estimator = grid_ml.best_estimator_
    X_test_std = sc.transform(X_test)
    y_pred = estimator.predict(X_test_std)
    print("테스트 데이터 세트 정확도: {0:.4f}".format(accuracy_score(y_test, y_pred)))
    #학습이 완료된 객체를 지정한다. 
    with open('./7.Scikit-Tree/ml.dat', 'wb') as fp:
        pickle.dump(sc,fp)
        pickle.dump(grid_ml,fp)
    print('학습 완료')

    # 시각화를 위한 작업 
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train, y_test))
    plot_decision_region(X=X_combined_std, y=y_combined_std, classifier=grid_ml, test_idx = range(70, 100), title="perceptron")

def step3_using():
    #(학습완료된 객체 복원)
    with open("./7.Scikit-Tree/ml.dat", "rb") as fp :
        sc = pickle.load(fp)
        ml = pickle.load(fp)

    X = [
        [1.4, 0.2],[1.3, 0.2], [1.5, 0.2],
        [4.5, 1.5],[4.1, 1.0], [4.5, 1.5],
        [5.2, 2.0],[5.4, 2.3], [5.1, 1.8]

    ]
    X_std = sc.transform(X)

    #결과를 추출한다
    y_pred = ml.predict(X_std)

    for value in y_pred : 
        if value == 0 : 
            print("Iris-setosa")
        elif value == 1 :
            print('Iris - versicolor')
        elif value == 2:
            print("Iris-virginica")
    
if __name__ == "__main__":
    step1_get_data()
    step2_learning()
    step3_using()