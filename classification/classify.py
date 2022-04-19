# encoding:utf-8
import json
from sklearn.model_selection import train_test_split
from bert_serving.client import BertClient
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.externals import joblib
import classification.process

def gbm_classific(x,y):
    '''
        分类器lightgbm进行分类
        :return model: 得到的分类器
        '''

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
    joblib.dump(model, 'actorscope.pkl')

    y_pred_pa = model.predict(x_test)  # 输出类别
    result = y_pred_pa.argmax(axis=1)
    print(classification_report(y_test, result))
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
    # bc = BertClient(port=6791, port_out=6792, check_length=False)
    bc = BertClient(ip="60.205.188.102", port=16002, port_out=16003, check_length=False, check_version=False)
    dec = bc.encode(word_v)
    return dec



def get_data():
    x = []
    y = []
    count_label=0
    value_list = []
    title_list=[]
    file1 = open('data/labeled_test.txt', 'r', encoding='utf8').readlines()
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
    return x,y

value_json={"0": "",
            "1": "PRV",
            "2": "INV",
            "3": "RUV",
            "4": "TIV",
            "5": "EJV",
            "6": "KSV",
            "7": "MIV",
            "8": "ECV",
            "9": "CAV",
            "10": "PXV"
}

def get_scope_classify(file_name, node_g):
    graph = { "node" :[],
              "rela" :[]
    }
    node_list = []
    n_list = []
    # file = open(file_name, 'r', encoding='utf8').readlines()
    df = file_name.read().decode()
    a = df.splitlines()
    if node_g is None or node_g == []:
        for line in a:
            line = json.loads(line)
            actor, value = classification.process.read_clean(line)
            if "" in value:
                value.remove("")
            x = get_embedding(value)
            model = joblib.load('classification/actorscope.pkl')
            y_pred_pa = model.predict(x, num_iteration=model.best_iteration)  # 输出类别
            result = y_pred_pa.argmax(axis=1)
            node1 = {
                "name": actor,
                "type": "",
                "ntype": "actor"
            }
            graph["node"].append(node1)
            for lab, v in zip(result, value):
                if lab != 0:
                    node3 = {
                        "name": v,
                        "type": value_json[str(lab)],
                        "ntype": "value"
                    }
                    if node3 not in graph["node"]:
                        graph["node"].append(node3)
                    rel1 = {
                        "source": actor,
                        "target": v,
                        "r": "A-V"
                    }
                    if rel1 not in graph["rela"]:
                        graph["rela"].append(rel1)
    else:
        for n in node_g:
            n_list.append(n["name"])
        for line in file:
            line = json.loads(line)
            actor, value = classification.process.read_clean(line)
            for node in node_g:
                if actor in node["name"] or node["name"] in actor:
                    x = get_embedding(value)
                    model = joblib.load('actorscope.pkl')
                    y_pred_pa = model.predict(x, num_iteration=model.best_iteration)  # 输出类别
                    result = y_pred_pa.argmax(axis=1)
                    node1 = {
                        "name": actor,
                        "type": "",
                        "ntype": "actor"
                    }
                    graph["node"].append(node1)
                    for lab, v in zip(result, value):
                        node3 = {
                            "name": v,
                            "type": value_json[str(lab)],
                            "ntype": "value"
                        }
                        if node3 not in graph["node"]:
                            graph["node"].append(node3)
                        rel1 = {
                            "source": actor,
                            "target": v,
                            "r": "A-V"
                        }
                        if rel1 not in graph["rela"]:
                            graph["rela"].append(rel1)
    return graph




if __name__ == '__main__':
    # read_file()
    # 规定格式：{“actor”: "某某科技公司",
    #           “value”: "智能照明技术领域内的技术研发、技术咨询、技术服务、技术转让；计算机软件、硬件、通讯设备及配件、
    #           家居用品、装饰用品、照明用品、电气设备、电工电料、电器、五金产品、电光源、照明器具、电器开关、家用电器、
    #           厨房卫浴洁具、家具家纺、智能家居产品的研发、技术转让、销售及安装服务；照明系统的研发、设计、销售及咨询服务；
    #           货物与技术进出口业务（法律行政法规禁止类项目不得经营，法律行政法规限制类项目许可后经营）。
    #           （依法须经批准的项目，经相关部门批准后方可开展经营活动）"}，
    #           {“actor”: "某某广告公司",
    #           “value”: "文化艺术咨询服务；广告设计、制作服务、国内外代理服务、发布服务；工艺品、日用百货、
    #           服装、鞋帽的零售；美术图案设计服务；舞台表演艺术指导服务；"}，
    file = 'data/test.txt'
    node_g = []
    graph = get_scope_classify(file, node_g)
    print(graph)

# x_train_shape:(1218, 788)
# x_test_shape:(406, 788)
# 模型训练集得分：0.993
# 模型测试集得分：0.990
