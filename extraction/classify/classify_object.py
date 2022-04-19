
# encoding:utf-8
from bs4 import BeautifulSoup
import numpy as np
# import lightgbm as lgb
import json
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from bert_serving.client import BertClient
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import neighbors, datasets
import lightgbm as lgb
from sklearn.externals import joblib
# from sklearn.grid_search import GridSearchCV
import pandas as pd

def gbm_classific():
    '''
        分类器lightgbm进行分类
        :return model: 得到的分类器
        '''

    # dict_y = { 'N': 0,
    #          'P-S': 1,#'供应商',
    #          'PE': 2, #'营销平台',
    #          'CE-S': 3, # '线下售卖门店/体验门店',
    #          'CE-A': 4, #'售后服务商',
    #          'PE-P': 5,#'摄影拍摄/短视频制作/店铺装修',
    # }
    # y_trans=[]
    # for y in y:
    #     y_trans.append(dict_y[y])
    # print(y_trans)
    # 得到标注数据进行分类训练
    x, y = get_train_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        # 'objective': 'multiclass',# 用于分类binary multiclass
        # 'num_class': 11,
        'objective': 'binary',

        'silent': False,  # 是否打印信息，默认False

        'learning_rate': 0.05,  # 选定一较高的值，通常是0.1
        'num_leaves': 60,  # 由于lightGBM是leaves_wise生长，官方说法是要小于2^max_depth 50
        'max_depth': 6,  # 由于数据集不大，所以选择一个适中的值，4-10都可以

        'subsample': 0.8,  # 数据采样
        'colsample_bytree': 0.8,  # 特征采样
    }


    cv_results = lgb.cv(params, lgb_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
                        metrics='auc', early_stopping_rounds=50, seed=0)
    print('best n_estimators:', len(cv_results['auc-mean']))
    print('best cv score:', pd.Series(cv_results['auc-mean']).max())

    # data_train = lgb.Dataset(x_train, y_train, silent=True)
    # cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
    #                     metrics='multi_logloss', early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=2019)
    # print('best n_estimators:', len(cv_results['multi_logloss-mean']))
    # print('best cv score:', cv_results['multi_logloss-mean'][-1])

    # # 导出结果
    # for pred in preds:
    #     result = int(np.argmax(pred))

        # 导出特征重要性
    # importance = gbm.feature_importance()
    # names = gbm.feature_name()
    # with open('./feature_importance.txt', 'w+') as file:
    #     for index, im in enumerate(importance):
    #         string = names[index] + ', ' + str(im) + '\n'
    #         file.write(string)

    model = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=200
                      )
    # model.save_model('model.txt')
    # model = lgb.Booster(model_file='model.txt')  # init model

    # joblib.dump(model, 'object.pkl')
    print('Start predicting...')

    y_pred_pa = model.predict(x_test, num_iteration=model.best_iteration)  # 输出类别
    # result = y_pred_pa.argmax(axis=1)
    #二分类
    result = []
    threshold = 0.5
    for pred in y_pred_pa:
        result_ind = 1 if pred > threshold else 0
        result.append(result_ind)
    print(classification_report(y_test, result, digits=4))
    # model.fit(x, y)
    return model
    # result =model.predict(x)  # 输出类别

    # print(model.predict_proba(x_test) ) # 输出分类概率
    # model.predict_log_proba(x_test)  # 输出分类概率的对数


# 测试分类模型效果，调参
def svm_classific():
    x, y = get_train_data()
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

    print(classification_report(y_test, result, digits=4))
    return model

def mnb_classific():
    # 使用贝叶斯进行训练
    x, y = get_train_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    model = GaussianNB()
    model.fit(x_train, y_train)
    result= model.predict(x_test)
    print('模型训练集得分：{:.3f}'.format(
        model.score(x_train, y_train)))
    # 打印测试集得分
    print( '模型测试集得分：{:.3f}'.format(
        model.score(x_test, y_test)))
    print(classification_report(y_test, result, digits=4))
    return model


def knn_classific():
    x, y = get_train_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    model = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
    # KNN来拟合 选择K=15 权重为距离远近
    model.fit(x_train, y_train)
    result= model.predict(x_test)
    print(classification_report(y_test, result, digits=4))
    return model


def read_file():
    mapped_f = open("./json_data/5page_token_labeled.json", 'w', encoding='utf-8')
    for line in open("./json_data/5page_token.json", 'r', encoding='utf-8', errors='ignore').readlines():
        value = json.loads(line)
        mapped_f.write("0"+"\t"+ json.dumps(value, ensure_ascii=False) + '\n')


def get_embedding(word_v):
    '''
    利用Bert训练输入的词或句
    :param word_v: 输入想要得到向量的词
    :param bc: bert向量化
    :return dec: 得到的向量
    '''
    bc = BertClient(port=6789, port_out=6790, check_length=False)
    # bc = BertClient(ip="60.205.188.102", port=16002, port_out=16003, check_length=False, check_version=False)
    dec = bc.encode(word_v)
    return dec


def get_train_data():
    x = []
    y = []
    count_label=0
    token_list = []
    obj1 = []
    value_list = []
    title_list=[]
    # file1 = open('labeled_test.txt', 'r', encoding='utf8').readlines()
    file1 = open('./classify/labeled_test.txt', 'r', encoding='utf8').readlines()
    for line in file1:
        line_label = line.split('\t')
        #二分类
        if int(line_label[0])!= 0:
            line_label[0] = 1

        y.append(int(line_label[0]))
        if int(line_label[0])!= 0:
            count_label = count_label+1
        # x.append(list(map(float,list(sp[1].split(' ')))))
        line_value = line_label[1:]
        line_value = line_value[0].replace("\n", "")
        value_list.append(line_value)
    x = get_embedding(value_list)
    return x,y

def get_predict_data():
    x = []
    y = []
    count_label=0
    token_list = []
    obj = []
    value_list = []
    title_list=[]
    file1 = open('labeled.txt', 'r', encoding='utf8').readlines()
    for line in file1:
        line_label = line.split('\t')
        y.append(int(line_label[0]))
        if int(line_label[0])!= 0:
            count_label = count_label+1
        # x.append(list(map(float,list(sp[1].split(' ')))))
        line_value = line_label[1:]
        line_value = line_value[0].replace("\n", "")
        value_list.append(line_value)
    x = get_embedding(value_list)

    return x, value_list


def get_predict_object(obj):
    value_list = []
    for line in obj:
        value_list.append(line)
    x = get_embedding(value_list)

    return x, value_list


def predict_data(obj):
    '''
    利用分类器得到分类结果
    :param file_name: 输入html文件
    :return dicsen: 得到分类后的label和元素
    :return label_list: 得到分类后的label的种类
    '''
    # x, obj = get_predict_data()
    # 得到想预测的数据的向量和数据元素
    model = gbm_classific()
    # model = svm_classific()
    # model = mnb_classific()
    # model = knn_classific()
    x, obj = get_predict_object(obj)
    # model = joblib.load('extraction/classify/object.pkl')

    y_pred_pa = model.predict(x)  # 输出类别
    # y_pred_pa = model.predict(x, num_iteration=model.best_iteration)  # 输出类别
    # result = y_pred_pa.argmax(axis=1)
    #二分类
    result = []
    threshold = 0.2
    for pred in y_pred_pa:
        print(pred)
        result_ind = 1 if pred > threshold else 0
        result.append(result_ind)
    value_list=[]
    for label, text in zip(result, obj):
        value = {"label": str(label),
                 "object": str(text)}
        value_list.append(value)
        # print(str(label) + "\t" + str(text) )

    return value_list


    # file_write = open("labeled_all.txt", 'w', encoding='utf-8') #输出文件
    # obj_list = []
    # for label, text in zip(result,obj):
    #     if text not in obj_list:
    #         obj_list.append(text)
    #         file_write.write(str(label) + "\t" + str(text) + '\n')

def classify(obj):
    predict_data()


if __name__ == '__main__':
    # read_file()
    # x, y = get_data()
    # mnb_classific(x, y)
    predict_data()
    # train_data()

# json={"0": none;
# "1":{PRV}{产品类价值}-产品所有权
# "2":{INV}{信息类价值}-信息
# "3":{RUV}{资源使用类价值}-软件、设备、人力资源使用权
# "4":{TIV}{事物状态改变类价值}-物理对象的状态
# "5":{EJV}{享受类价值}-用户精神状态
# "6":{KSV}{知识与技能类价值}-知识技能
# "7":{MIV}{市场影响类价值}-市场影响力
# "8":{ECV}经济类价值-金钱数量
# "9":{CAV}用户聚集类价值-用户需求信息
# "10":{PXV}经验类价值-经验
# }