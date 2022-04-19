import numpy as np
import pandas as pd
from collections import Counter
from bert_serving.client import BertClient
from sklearn.cluster import KMeans
from numpy.testing import rundocs
import json
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch


def get_token_vector(token_list):
    embed_token_list = []
    EMBED_TOKEN_LENGTH = 20
    # html_label = ['html', 'head', 'meta', 'title', 'body', 'div', 'a', 'span', 'i', 'ul', 'li', 'p', 'label', 'h1',
    #               'h2',
    #               'h3', 'h4', 'h5', 'h6', 'dl', 'dt', 'dd', 'section', 'header', 'nav']
    html_label_dic = {'html': 0, 'head': 1, 'meta': 2, 'title': 3, 'body': 4, 'div': 5, 'a': 6, 'span': 7, 'i': 8,
                      'ul': 9, 'li': 10, 'p': 11, 'label': 12, 'h1': 13, 'h2': 14, 'h3': 15, 'h4': 16, 'h5': 17,
                      'h6': 18, 'dl': 19, 'dt': 20, 'dd': 21, 'section': 22, 'header': 23, 'nav': 24}
    for token in token_list:
        embed_token = [0] * EMBED_TOKEN_LENGTH
        for count, html_label in enumerate(token):
            if count < (EMBED_TOKEN_LENGTH-1):
                html_value = html_label_dic.get(html_label, None)
                if html_value is None:
                    embed_token[count] = 25
                else:
                    embed_token[count] = html_value
        embed_token_list.append(embed_token)
    embed_token_list = np.array(embed_token_list)  # 列表转数组
    return embed_token_list


def get_embedding(word_v, bc):
    '''
    利用Bert训练输入的词或句
    :param word_v: 输入想要得到向量的词
    :param bc: bert向量化
    :return dec: 得到的向量
    '''
    dec = bc.encode(word_v)
    return dec


def kmeans_cluster(n_clusters,dec):
    '''
    利用Kmeans聚类输入的向量
    :param dec: 输入需要训练的向量
    :return kmeans: 得到聚类后的结果
    '''
    #     kmeans = KMeans(n_clusters=10,n_init=50, init='k-means++')
    kmeans = KMeans(n_clusters)
    y = kmeans.fit_predict(dec)
    return y, kmeans


def gmm_cluster(n_clusters,c):
    '''
    利用gmm高斯 聚类输入的向量
    :param c: 输入需要训练的向量
    :return y_pred: 得到聚类后的label
    '''
    ##设置gmm函数
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(c)
    ##训练数据
    y_pred = gmm.predict(c)
    print(gmm.means_)
    return y_pred


def ac_cluster(n_clusters,c):
    '''
    利用ac层次聚类 输入的向量
    :param c: 输入需要训练的向量
    :return y_pred: 得到聚类后的label
    '''
    linkages = ['ward', 'average', 'complete']
    # linkages为ac聚类中求距离的方式，这里尝试最小距离
    # ward：组间距离等于两类对象之间的最小距离。（即single - linkage聚类）
    # average：组间距离等于两组对象之间的平均距离。（average - linkage聚类）
    # complete：组间距离等于两组对象之间的最大距离。（complete - linkage聚类）
    ac = AgglomerativeClustering(linkage=linkages[0], n_clusters=n_clusters)
    ##训练数据
    ac.fit(c)
    ##每个数据的分类
    y_pred = ac.labels_
    return y_pred


def birch_cluster(n_clusters, c):
    '''
    利用birch聚类特征树 聚类输入的向量
    :param c: 输入需要训练的向量
    :return y_pred: 得到聚类后的label
    '''
    birch = Birch(n_clusters=n_clusters)
    ##训练数据
    y_pred = birch.fit_predict(c)
    return y_pred


def dbscan_cluster(c):
    #     待完成
    dbscan = DBSCAN(eps=0.4, min_samples=9)
    dbscan.fit(c)
    y_pred = dbscan.labels_


def get_2d_embedding(token_list, value_list):
    bc = BertClient(port=6791, port_out=6792)
    embed_token_list = get_token_vector(token_list)
    embedding_value = get_embedding(value_list, bc)
    c = np.hstack((embed_token_list, embedding_value))
    return c


if __name__ == '__main__':
    token_list = []
    obj1 = []
    value_list = []
    # title_list=[]
    file1 = open('./token_final.json', 'r', encoding='utf8').readlines()
    for line in file1:
        value = json.loads(line)
        obj1.append(value)
        value_list.append(value["value"])
        token_list.append(value["token"])
        # title_list.append(value["title"])

    bc = BertClient(port=6791, port_out=6792)
    embed_token_list = get_token_vector(token_list)
    embedding_value=get_embedding(value_list, bc)
    # embedding_title=get_embedding(title_list, bc)

    c = np.hstack((embed_token_list, embedding_value))
    wb = open('label.txt', 'w', encoding='utf-8')
    for l in c:
        print(l)
        wb.write("0,"+l)
        wb.write('\n')
        # wb.close()
    # c = np.hstack((c, embedding_title))
    # c=c.reshape(-1, 1)
    # n_clusters = 15
    # y_pred, kmeans11 = kmeans_cluster(n_clusters, c)
    #
    # dicsen = {}
    # for index, label in enumerate(y_pred, 1):
    #     #     print("index: {}, label: {}".format(index, label))
    #     dicsen.setdefault(label, []).append(obj1[index - 1])
    #
    # wf_f1 = open('k_3diam.txt', 'w', encoding='utf8')
    # for label in range(15):
    #     wf_f1.write("\n分类%d：\n" % label)
    #     word_frequent = []
    #     for sen in dicsen[label]:
    #         wf_f1.write(str(sen) + "\n")
    #         word_frequent.append(sen["value"])
    #     print(word_frequent)

# def kmeans():
#     y_pred, kmeans11 = get_kmeans(c)
#     print("inertia: {}".format(kmeans11.inertia_))
#     print(kmeans11.labels_)

#     # wordvector = []
#     dicsen11 = {}
#     for index, label in enumerate(kmeans11.labels_, 1):
#     #     print("index: {}, label: {}".format(index, label))
#         dicsen11.setdefault(label, []).append(obj1[index - 1])

#     wf_f1 = open('3page_result_value_2diam.txt', 'w', encoding='utf8')
#     for label in range(15):
#         wf_f1.write("\n分类%d：\n" % label)
#         word_frequent11 = []
#         for sen in dicsen11[label]:
#             wf_f1.write(str(sen) + "\n")
#             word_frequent11.append(sen["value"])
#         print(word_frequent11)