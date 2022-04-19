import json
from matplotlib.colors import rgb2hex
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from bert_serving.client import BertClient
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from html.parser import HTMLParser

def embed_value():
    obj = []
    value_list = []
    file = open('./全国小米之家_token_final.json', 'r', encoding='utf8').readlines()
    for line in file:
        value = json.loads(line)
        obj.append(value)
        value_list.append(value["value"])

    bc = BertClient(port=6789, port_out=6790)
    embedding1 = bc.encode(value_list)

    kmeans1 = KMeans(n_clusters=15)
    # kmeans = KMeans(n_clusters=10,n_init=50, init='k-means++')
    y_pred = kmeans1.fit_predict(embedding1)
    print("inertia: {}".format(kmeans1.inertia_))
    print(kmeans1.cluster_centers_)
    # wordvector = []
    dicsen1 = {}
    for index, label in enumerate(kmeans1.labels_, 1):
        #     print("index: {}, label: {}".format(index, label))
        dicsen1.setdefault(label, []).append(obj[index - 1])

    wf_f = open('result_value.txt', 'w', encoding='utf8')
    for label in range(15):
        wf_f.write("\n分类%d：\n" % label)
        word_frequent1 = []
        for sen in dicsen1[label]:
            wf_f.write(str(sen)+"\n" )
            word_frequent1.append(sen["value"])
        print(word_frequent1)


def show_cluster(embedding,y_pred):
    # 以下为示意，train为训练数据
    clusters_number = 15
    # y_pred = KMeans(n_clusters=clusters_number, random_state=9).fit_predict(train)
    train = embedding
    tsne = TSNE(n_components=2)
    train = tsne.fit_transform(train)
    fig, ax = plt.subplots()
    types = []
    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(clusters_number)])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

    for i, color in enumerate(colors):
        need_idx = np.where(y_pred == i)[0]
        ax.scatter(train[need_idx, 1], train[need_idx, 0], c=color, label=i)

    plt.figure(figsize=(15, 15))
    # 改变坐标轴间隔
    # x_locator = MultipleLocator(0.01)
    # y_locator = MultipleLocator(0.01)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_locator)
    # ax.yaxis.set_major_locator(y_locator)

    legend = ax.legend(loc='upper right')
    plt.show()

def common_token(input_file):
    tagstack = []
    maxlen = 1
    for line in open(input_file, 'r', encoding='utf-8', errors='ignore').readlines():
        value = json.loads(line)
        token=value["token"]
        len_t=len(token)
        if len_t > maxlen:
            maxlen=len_t
        for t in token:
            if t not  in tagstack:
                tagstack.append(t)
    # print(maxlen) 17
# common_token("token_final.json")
# ['html', 'body', 'div', 'a', 'span', 'i', 'ul', 'li', 'p', 'label', 'h1', 'h3', 'h2', 'h4', 'dl', 'dt', 'dd', 'head', 'meta', 'title', 'section', 'header', 'h5', 'h6', 'nav']

def show_sse(embedding):
    SSE = []  # 存放每次结果的误差平方和
    for k in range(5,20):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(embedding)
        SSE.append(estimator.inertia_)
    X = range(5,20)
    plt.xlabel('簇的个数')
    plt.ylabel('簇内离差平方和之和')
    plt.plot(X,SSE,'o-')
    plt.show()

def show_silhouette(e):
    Scores = []  # 存放轮廓系数
    for k in range(5,20):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit_predict(e)
        Scores.append(silhouette_score(e,estimator.labels_,metric='euclidean'))
    X = range(5,20)
    plt.xlabel('k')
    plt.ylabel('轮廓系数')
    plt.plot(X,Scores,'o-')
    plt.show()

html_label = ['html', 'head', 'meta', 'title', 'body', 'div', 'a', 'span', 'i', 'ul', 'li', 'p', 'label', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dl', 'dt', 'dd',  'section', 'header',  'nav']
# print(len(html_label))  25

obj = []
token_list = []
embed_token_list = []
file = open('token_final.json', 'r', encoding='utf8').readlines()
for line in file:
    value = json.loads(line)
    obj.append(value)
    token_list.append(value["token"])
for token in token_list:
    embed_token = [0 for x in range(0, 20)]
    for count, label in enumerate(token):
        if count < 19:
            label_in_list = False
            for i, j in enumerate(html_label):
                if label == j:
                    label_in_list = True
                    embed_token[count] = i
            if label_in_list == False:
                embed_token[count] = 25
    embed_token_list.append(embed_token)
print(embed_token_list)
embed_token_list = np.array(embed_token_list)    # 列表转数组
print(embed_token_list, end='\n\n')
print (embed_token_list.ndim)

# len_token=len(obj)
# for i in range(len_token):
#     c = np.hstack((a,b))
