# encoding:utf-8
from bs4 import BeautifulSoup
import numpy as np
# import lightgbm as lgb
import json
import cluster
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from bert_serving.client import BertClient
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import neighbors, datasets


# 测试分类模型效果，调参
def get_svm_classific():
    '''
    分类器进行分类
    :return model: 得到的分类器
    '''
    x, y = get_train_data()
    # 得到标注数据进行分类训练
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

    # model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
    # tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)

    # model = svm.SVC( C = 1.0, cache_size = 200, class_weight = None, coef0 = 0.0,
    # decision_function_shape = 'ovo', degree = 3, gamma = 'auto', kernel = 'rbf',
    # max_iter = -1, probability = False, random_state = None, shrinking = True,
    # tol = 0.001, verbose = False)
    #
    # model = GaussianNB()

    model = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')

    model.fit(x_train, y_train)
    result =model.predict(x_test)  # 输出类别
    print(classification_report(y_test, result))
    model.fit(x, y)
    print('模型训练集得分：{:.3f}'.format(
        model.score(x_train, y_train)))
    # 打印测试集得分
    print( '模型测试集得分：{:.3f}'.format(
        model.score(x_test, y_test)))
    return model
    # result =model.predict(x)  # 输出类别

    # print(model.predict_proba(x_test) ) # 输出分类概率
    # model.predict_log_proba(x_test)  # 输出分类概率的对数


def svm_classific(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    print('x_train_shape:{}'.format(x_train.shape))
    print('x_test_shape:{}'.format(x_test.shape))
    # print('y_train_shape:{}'.format(y_train.shape))
    # print('y_test_shape:{}'.format(y_test.shape))

    # model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
    # tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)
    model = svm.SVC( C = 1.0, cache_size = 200, class_weight = None, coef0 = 0.0,
    decision_function_shape = 'ovo', degree = 3, gamma = 'auto', kernel = 'rbf',
    max_iter = -1, probability = False, random_state = None, shrinking = True,
    tol = 0.001, verbose = False)

    # clf = svm.SVC(probability = True)
    # clf = svm.SVR()
    # clf.fit(x,y)
    model.fit(x_train, y_train)
    result =model.predict(x_test)  # 输出类别
    # print(result)
    # model.predict_proba(x_test)  # 输出分类概率
    # model.predict_log_proba(x_test)  # 输出分类概率的对数

    print('模型训练集得分：{:.3f}'.format(
        model.score(x_train, y_train)))
    # 打印测试集得分
    print( '模型测试集得分：{:.3f}'.format(
        model.score(x_test, y_test)))

    print(classification_report(y_test, result))

def mnb_classific(x,y):
    # 使用贝叶斯进行训练
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    model = GaussianNB()
    model.fit(x_train, y_train)
    result= model.predict(x_test)
    print('模型训练集得分：{:.3f}'.format(
        model.score(x_train, y_train)))
    # 打印测试集得分
    print( '模型测试集得分：{:.3f}'.format(
        model.score(x_test, y_test)))
    print(classification_report(y_test, result))


def knn_classific(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    model = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
    # KNN来拟合 选择K=15 权重为距离远近
    model.fit(x_train, y_train)
    result= model.predict(x_test)
    print(classification_report(y_test, result))


def read_file():
    mapped_f = open("./json_data/5page_token_labeled.json", 'w', encoding='utf-8')
    for line in open("./json_data/5page_token.json", 'r', encoding='utf-8', errors='ignore').readlines():
        value = json.loads(line)
        mapped_f.write("0"+"\t"+ json.dumps(value, ensure_ascii=False) + '\n')


def get_data():
    x = []
    y = []
    count_label=0

    token_list = []
    obj1 = []
    value_list = []
    title_list=[]
    file1 = open('./token_labeled.json', 'r', encoding='utf8').readlines()
    for line in file1:
        line_label = line.split('\t')
        y.append(int(line_label[0]))
        if int(line_label[0])!= 0:
            count_label = count_label+1
        # x.append(list(map(float,list(sp[1].split(' ')))))
        line_value = line_label[1:]
        line_value = line_value[0].replace("\n", "")
        # print(line_value)
        value = json.loads(line_value)
        obj1.append(value)
        value_list.append(value["value"])
        token_list.append(value["token"])
        title_list.append(value["title"])

    bc = BertClient(port=6791, port_out=6792,check_length=False)
    embed_token_list = cluster.get_token_vector(token_list)
    embedding_value = cluster.get_embedding(value_list, bc)
    embedding_title=cluster.get_embedding(title_list, bc)

    x = np.hstack((embed_token_list, embedding_value))
    return x,y


if __name__ == '__main__':
    # read_file()
    x, y = get_data()
    mnb_classific(x, y)

# x_train_shape:(1218, 788)
# x_test_shape:(406, 788)
# 模型训练集得分：0.993
# 模型测试集得分：0.990

# 123类数据太少，全是0类



# from sklearn import tree, svm, naive_bayes, neighbors
# from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
# from sklearn.multiclass import OneVsRestClassifier
#
# from basic.CommunityFeature import *
# from community_alg.GED import GED
#
# SKLEARN_CLASSIFICATIONS = {
#     'svm': OneVsRestClassifier(svm.LinearSVC()),
#     'decision_tree': tree.DecisionTreeClassifier(),
#     'naive_gaussian': naive_bayes.GaussianNB(),
#     'naive_mul': naive_bayes.MultinomialNB(),
#     'K_neighbor': neighbors.KNeighborsClassifier(),
#     'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
#     'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
#     'random_forest': RandomForestClassifier(n_estimators=50),
#     'adaboost': AdaBoostClassifier(n_estimators=50),
#     'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
# }

