# encoding:utf-8
from bs4 import BeautifulSoup
import numpy as np
# import lightgbm as lgb
import json
import ast_html.cluster as cluster
import ast_html.preprocess
from bert_serving.client import BertClient
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import neighbors, datasets
import os
import lightgbm as lgb
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


# 分类结果

def gbm_classific():
    '''
        分类器lightgbm进行分类
        :return model: 得到的分类器
        '''

    x, y = get_train_data()
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',  # 用于分类
        'num_class': 11,
        'silent': False,  # 是否打印信息，默认False

        'learning_rate': 0.1,  # 选定一较高的值，通常是0.1
        'num_leaves': 50,  # 由于lightGBM是leaves_wise生长，官方说法是要小于2^max_depth
        'max_depth': 6,  # 由于数据集不大，所以选择一个适中的值，4-10都可以

        'subsample': 0.8,  # 数据采样
        'colsample_bytree': 0.8,  # 特征采样
    }
    model = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=500)

    print('Start predicting...')
    # joblib.dump(model, 'htmlactor.pkl')

    y_pred_pa = model.predict(x_test)  # 输出类别
    result = y_pred_pa.argmax(axis=1)
    # print(result)
    # print(y_test)
    print(classification_report(y_test, result))
    # model.fit(x, y)
    return model
    # result =model.predict(x)  # 输出类别

    print(model.predict_proba(x_test) ) # 输出分类概率
    model.predict_log_proba(x_test)  # 输出分类概率的对数

def gbm_classific_expriment():
    '''
        做实验
        :return model: 得到的分类器
        '''

    x, y = get_train_data_expriment()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',  # 用于分类
        'num_class': 11,
        'silent': False,  # 是否打印信息，默认False
        'learning_rate': 0.1,  # 选定一较高的值，通常是0.1
        'num_leaves': 50,  # 由于lightGBM是leaves_wise生长，官方说法是要小于2^max_depth
        'max_depth': 6,  # 由于数据集不大，所以选择一个适中的值，4-10都可以
        'subsample': 0.8,  # 数据采样
        'colsample_bytree': 0.8,  # 特征采样
    }
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=10000,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=500)
    # model = joblib.load('./htmlactor.pkl')
    print('Start predicting...')
    # joblib.dump(model, 'htmlactor.pkl')

    y_pred_pa = model.predict(x_test)  # 输出类别
    result = y_pred_pa.argmax(axis=1)

    print(classification_report(y_test, result, digits=4))
    # model.fit(x, y)
    # result =model.predict(x)  # 输出类别

    # print(model.predict_proba(x_test) ) # 输出分类概率
    # print(model.predict_log_proba(x_test))  # 输出分类概率的对数

def train_data(file_name):
    '''
    预测 利用分类器得到分类结果
    :param file_name: 输入html文件
    :return dicsen: 得到分类后的label和元素
    :return label_list: 得到分类后的label的种类
    '''
    x, obj = get_predict_data(file_name)
    # 得到想预测的数据的向量和数据元素

    model = gbm_classific()
    y_pred_pa = model.predict(x)  # 输出类别

    # model = joblib.load('ast_html/htmlactor.pkl')
    # model = joblib.load('./htmlactor.pkl')
    # y_pred_pa = model.predict(x, num_iteration=model.best_iteration)  # 输出类别
    result = y_pred_pa.argmax(axis=1)
    label_list=[]
    for res in result:
        if res not in label_list:
            label_list.append(res)
    label_list.remove(0)
    dicsen = {}
    dicsen_cate = {}
    for index, label in enumerate(result, 1):
        #     print("index: {}, label: {}".format(index, label))
        dicsen.setdefault(label, []).append(obj[index - 1])
    for label in label_list:
        word_frequent1 = []
        for sen in dicsen[label]:
            word_frequent1.append(sen["value"])
        dicsen_cate.setdefault(label, word_frequent1)
    return dicsen_cate, label_list

def get_train_data_expriment():
    '''
    实验训练部分
    :return x: 得到训练后的向量
    :return y: 得到分类后的标签
    '''
    x = []
    y = []
    count_label=0

    token_list = []
    obj1 = []
    value_list = []
    title_list = []
    keyword_list = []
    context_list = []
    file1 = open('./html_labeled_0428updata.json', 'r', encoding='utf8').readlines()
    for line in file1:
        line_label = line.split('\t')
        y.append(int(line_label[0]))
        if (line_label[0])!= "0":
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
        if value["keyword"] == "":
            keyword_list.append(value["title"])
        else:
            keyword_list.append(value["keyword"])
        if value["context"] == "":
            context_list.append("title")
        else:
            context_list.append(value["context"])
    print(count_label)

    # bc = BertClient(ip="60.205.188.102", port=16002, port_out=16003, check_length=False, check_version=False)

    bc = BertClient( port=6789, port_out=6790, check_length=False)
    embed_token_list = cluster.get_token_vector(token_list)
    embedding_value = cluster.get_embedding(value_list, bc)
    embedding_title = cluster.get_embedding(title_list, bc)
    embedding_keyword = cluster.get_embedding(keyword_list, bc)
    embedding_context = cluster.get_embedding(context_list, bc)

    # x = np.hstack((embed_token_list, embedding_value, embedding_title, embedding_keyword, embedding_context))
    # x= np.hstack((embedding_title, embedding_value))
    # x = np.hstack((embedding_value, embedding_title, embedding_keyword, embedding_context))
    x = embedding_value
    # x = embed_token_list
    return x, y


def get_train_data():
    '''
    并不执行- 训练 标注后的数据
    :return x: 得到训练后的向量
    :return y: 得到分类后的标签
    '''
    x = []
    y = []
    count_label=0

    token_list = []
    obj1 = []
    value_list = []
    title_list = []
    keyword_list = []
    context_list = []
    file1 = open('./html_labeled_0428updata.json', 'r', encoding='utf8').readlines()
    for line in file1:
        line_label = line.split('\t')
        y.append(int(line_label[0]))
        if (line_label[0])!= "0":
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
        if value["keyword"] == "":
            keyword_list.append(value["title"])
        else:
            keyword_list.append(value["keyword"])
        if value["context"] == "":
            context_list.append("title")
        else:
            context_list.append(value["context"])
    print(count_label)

    # bc = BertClient(ip="60.205.188.102", port=16002, port_out=16003, check_length=False, check_version=False)

    bc = BertClient( port=6789, port_out=6790, check_length=False)
    embed_token_list = cluster.get_token_vector(token_list)
    embedding_value = cluster.get_embedding(value_list, bc)
    embedding_title = cluster.get_embedding(title_list, bc)
    embedding_keyword = cluster.get_embedding(keyword_list, bc)
    embedding_context = cluster.get_embedding(context_list, bc)

    x = np.hstack((embed_token_list, embedding_value, embedding_title, embedding_keyword, embedding_context))
    # x= np.hstack((embed_token_list, embedding_value))
    return x, y

def get_predict_data(file_name):
    '''
    预测 训练向量输出给分类器
    :param file_name: 输入html文件
    :return x: 想要分类的元素的向量
    :return obj: 想要分类的元素的文本
    '''
    # x是数据，y是类别标签
    x = []
    token_list = []
    obj = []
    value_list = []
    title_list = []
    keyword_list = []
    context_list = []
    html_token_list = ast_html.preprocess.input_process_html(file_name)
    for line_value in html_token_list:
        # value = json.loads(line_value)
        obj.append(line_value)
        value = line_value
        value_list.append(value["value"])
        token_list.append(value["token"])
        title_list.append(value["title"])
        if value["keyword"] == "":
            keyword_list.append(value["title"])
        else:
            keyword_list.append(value["keyword"])
        if value["context"] == "":
            context_list.append("title")
        else:
            context_list.append(value["context"])
    bc = BertClient(port=6791, port_out=6792, check_length=False)
    # bc = BertClient(ip="60.205.188.102", port=16002, port_out=16003, check_length=False, check_version=False)
    embed_token_list = ast_html.cluster.get_token_vector(token_list)
    embedding_value = ast_html.cluster.get_embedding(value_list, bc)
    embedding_title = ast_html.cluster.get_embedding(title_list, bc)
    embedding_keyword = ast_html.cluster.get_embedding(keyword_list, bc)
    embedding_context = ast_html.cluster.get_embedding(context_list, bc)

    x = np.hstack((embed_token_list, embedding_value, embedding_title, embedding_keyword, embedding_context))
    # x = np.hstack(x,embedding_title)
    return x, obj


cate_dic = {
             1: 'Provider',
             2: 'PE',
             3: 'CE-S',
             4: 'CE-A',
             5: 'PE-P',
             6: 'P-T',
             7: 'P-A',
             8: 'P-S',
             9: 'P-I',}

# 对应type：{
#                  'Provider': '供应商''卖家',
#                  'PE': '营销平台',
#                  'CE-S': '线下售卖门店/体验门店',
#                  'CE-A': '售后服务商',
#                  'PE-P': '摄影拍摄/短视频制作/店铺装修',
#                  'P-T': '出行公司',
#                  'P-A': '住宿',
#                  'P-S': '景区',
#                  'P-I': '保险',}


def gene_graph(broker, dicsen_all, label_list_all):
    node_list = []
    for label in label_list_all:
        for text in dicsen_all[label]:
            type = cate_dic[label]
            name = text
            # ["Customer", "Broker", "Provider", "C-enabler", "P-enabler"]
            if type.startswith("PE"):
                actor_type = "P-enabler"
            elif type.startswith("CE"):
                actor_type = "C-enabler"
            else :
                actor_type = "Provider"
            node = {
                "name": name,
                "type": type,
                "actorType": actor_type,
                "ntype": "actor"
            }
            node_list.append(node)
    bro = {
                "name": broker,
                "type": "Broker",
                "actorType": "Broker",
                "ntype": "actor"
            }
    node_list.append(bro)
    cus = {
                "name": "顾客",
                "type": "Customer",
                "actorType": "Customer",
                "ntype": "actor"
            }
    node_list.append(cus)
    return node_list


#
# def get_html_classify(broker, path):
#     '''
#     方法一和方法二的step1
#     预测结果 利用分类器得到分类结果
#     :param file_name: 输入html文件
#     :return dicsen_all: 得到分类后的label和元素
#     :return label_list_all: 得到分类后的label类型
#     '''
#     file_list = []
#     dicsen_all = {}
#     label_list_all = []  #所有出现的类型label
#     # args = get_args_parser().parse_args(['-model_dir', '../bert/',
#     #                                      '-port', '6791',
#     #                                      '-port_out', '6792',
#     #                                      # '-max_seq_len', 'NONE',
#     #                                      # '-mask_cls_sep',
#     #                                      # '-cpu',
#     #                                      '-num_worker', '1',
#     #                                      # '-timeout', '-1',
#     #                                      # '-check_length', 'False'
#     #                                      ])
#     # server = BertServer(args)
#     # server.start()
#     for fpathe, dirs, fs in os.walk(path):
#         for f in fs:
#             file_list.append(f)
#     for file_name in file_list:
#         file = os.path.join(path, file_name)
#         dicsen, label_list = train_data(file)
#
#         for label in label_list:
#             if label not in label_list_all:
#                 label_list_all.append(label)
#         for label in label_list:
#             for text in dicsen[label]:
#                 dicsen_all.setdefault(label, []).append(text)
#     graph = gene_graph(broker, dicsen_all, label_list_all)
#     # BertServer.shutdown(port=16002)
#     # server.close()
#
#     return graph

def html_classify(broker, file_name):
    '''
    方法一和方法二的step1
    预测结果 利用分类器得到分类结果
    :param file_name: 输入html文件
    :return dicsen_all: 得到分类后的label和元素
    :return label_list_all: 得到分类后的label类型
    '''
    file_list = []
    dicsen_all = {}
    label_list_all = []  #所有出现的类型label

    for file in file_name:
        dicsen, label_list = train_data(file)

        for label in label_list:
            if label not in label_list_all:
                label_list_all.append(label)
        for label in label_list:
            for text in dicsen[label]:
                dicsen_all.setdefault(label, []).append(text)
    graph = gene_graph(broker, dicsen_all, label_list_all)
    # BertServer.shutdown(port=16002)
    # server.close()
    print(graph)

    return graph

if __name__ == '__main__':
    # read_file()
    # x, y = get_data()
    # mnb_classific(x, y)

    file_name = './html/小米授权服务中心网点 -沈阳小米商城.html'
    train_data(file_name)

    broker = "小米"
    path = './html'
    graph = html_classify(broker, path)
    print(graph)
    # gbm_classific_expriment()
