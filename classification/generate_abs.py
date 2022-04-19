import pandas as pd
import json
import copy

def get_abs_rel():
    rel_list = []
    f = open("data/abs4.1.txt", "r", encoding="utf-8")
    data = json.load(f)
    output = open("data/abs.txt", "w", encoding="utf-8")
    for d in data["dependencies"]:
        text = d["text"]
        s_id = d["source"]
        t_id = d["target"]
        for n in data["actors"]:
            if n["id"]==s_id:
                source = n["text"]
            if n["id"]==t_id:
                target = n["text"]
        rel={"s": source,
             "t": target,
             "r": text}
        rel_list.append(rel)
    for r in rel_list:
        output.write(json.dumps(r, ensure_ascii=False) + '\n')

# labeled_writer.write(json.dumps(encode_line, ensure_ascii=False) + '\n')
#读取抽象层节点和边的关系
# rel1 = {
#     "source": source,
#     "target": edg["r"],
#     "text": "PRV"
# }

def change_abs():
    file_name = "data/abstract.txt"
    f = open(file_name, "r", encoding="utf-8").readlines()
    for line in f :
        line = line.replace("\'", "\"")
        abs = json.loads(line)
        if "经济" in abs["r"]:
            abs["r"]="ECV"
        if "产品" in abs["r"]:
            abs["r"]="PRV"
        if "信息" in abs["r"]:
            abs["r"]="INV"
        if "资源" in abs["r"]:
            abs["r"]="RUV"
        if "市场" in abs["r"]:
            abs["r"]="MIV"
        if "淘宝" in abs["s"] or "平台" in abs["s"] or "携程" in abs["s"]:
            abs["s"]="Broker"
        if "线下售卖门店" in abs["s"]:
            abs["s"]="CE-S"
        if "景区" in abs["s"]:
            abs["s"]="P-S"
        if "摄影拍摄" in abs["s"]:
            abs["s"]="PE-P"
        if "保险" in abs["s"]:
            abs["s"]="P-I"
        if "供应商" in abs["s"] or "卖家" in abs["s"]:
            abs["s"]="Provider"
        if "出行" in abs["s"]:
            abs["s"]="P-T"
        if "售后" in abs["s"]:
            abs["s"]="CE-A"
        if "旅行社" in abs["s"]:
            abs["s"]="Provider"
        if "航空公司" in abs["s"]:
            abs["s"]="P-T"
        if "淘宝" in abs["t"] or "平台" in abs["t"] or "携程" in abs["t"]:
            abs["t"]="Broker"
        if "线下售卖门店" in abs["t"]:
            abs["t"]="CE-S"
        if "顾客" in abs["t"]:
            abs["t"]="Customer"
        if "顾客" in abs["s"]:
            abs["s"] = "Customer"
        if "售后" in abs["t"]:
            abs["t"]="CE-A"
        if "旅行社" in abs["t"]:
            abs["t"]="Provider"
        if "航空公司" in abs["t"]:
            abs["t"]="P-T"

# { ' Broker ': '中介',
# 				   'Customer ': '顾客‘’,
#                  ' Provider': '供应商''卖家',
#                  'PE': '营销平台',
#                  'CE-S': '线下售卖门店/体验门店',
#                  'CE-A': '售后服务商',
#                  'PE-P': '摄影拍摄/短视频制作/店铺装修',
#                  'P-T': '出行公司',
#                  'P-A': '住宿',
#                  'P-S': '景区',
#                  'P-I': '保险',}
def gene_value(graph_last, abs):
    # 输入读取节点graph_last
    # abs为另一文件中储存的抽象层关系
    abs_list = []
    df = abs.read().decode()
    a = df.splitlines()
    for line in a:
        line = line.replace("\'", "\"")
        abs = json.loads(line)
        abs_list.append(abs)
    graph_last = graph_last.replace("\'", "\"")
    graph_last = json.loads(graph_last)
    cate_list = []
    graph_new = {}
    graph_new = copy.deepcopy(graph_last)

    for r in graph_last["links"]:
        s = r["source"]
        v = r["target"]
        for n in graph_last["node"]:
            if n["ntype"] == "actor" and n["name"] == s:
                s_t = n["type"]
            if n["ntype"] == "value" and n["name"] == v:
                v_t = n["type"]
        r["type"] = [s_t, v_t]

    for n in graph_last["actors"]:
        type = n["customProperties"]["Atype"]
        if type not in cate_list:
            cate_list.append(type)
            for abstract in abs_list:
                if abstract["t"] == type:
                    s = abstract["s"]
                    v = abstract["r"]
                    for r in graph_last["rela"]:
                        print([s, v])
                        if r["type"] == [s, v]:
                            print([s, v])
                            sour = r["target"]
                            for n in graph_last["node"]:
                                if n["type"] == type:
                                    rela1 = {'source': sour,
                                            'target': n["name"],
                                            "r": "V-A",
                                            "type": [s, v]}
                                    graph_new['rela'].append(rela1)
    return graph_new


def process_pistar(pi):
    graph = {
        "actors": [],
        "relationship": []
    }
    for actor in pi["actors"]:
        node = {
            "text": actor["text"],
            "type": "istar.Actor",
            "ActorType": actor["ActorType"],
            "Atype": actor["customProperties"]["Atype"]
        }
        graph["actors"].append(node)
    for d in pi["links"]:
        s_id = d["source"]
        t_id = d["target"]
        for n in pi["actors"]:
            if n["id"] == s_id:
                source = {
                    "text": n["text"],
                    "type": "istar.Actor",
                    "ActorType": n["ActorType"],
                    "Atype": n["customProperties"]["Atype"]
                }
                source_type = n["customProperties"]["Atype"]
        for m in pi["orphans"]:
            if m["id"] == t_id:
                target = {
                    "text": m["text"],
                    "type": "istar.Value",
                    "Type": m["customProperties"]["Type"]
                }
                target_type = m["customProperties"]["Type"]
        rel = {"source": source,
               "target": target,
               "r": [source_type, target_type]}
        graph["relationship"].append(rel)
    return graph


def find_rel(graph_last):
    file_name_ab = "classification/data/abstract.txt"
    # 输入读取节点graph_last
    # abs为另一文件中储存的抽象层关系
    abs_list = []
    file = open(file_name_ab, 'r', encoding='utf8').readlines()
    for line in file:
        line = line.replace("\'", "\"")
        abs = json.loads(line)
        if abs not in abs_list:
            abs_list.append(abs)
    graph_last = json.loads(graph_last)
    print(graph_last)
    print(abs_list)
    graph_last = process_pistar(graph_last)
    cate_list = []
    graph_new = []
    print("开始循环")
    for n in graph_last["actors"]:
        type = n["Atype"]
        if type not in cate_list:
            cate_list.append(type)
            print("不同的类型target")
            print(type)
            for node_now in graph_last["actors"]:
                if node_now["Atype"] == type:
                    print("同类型遍历")
                    for abstract in abs_list:
                        print("抽象层现在是")
                        print(abstract)
                        if abstract["t"] == type:
                            print('现在抽象层t是')
                            print(abstract["t"])
                            s = abstract["s"]
                            v = abstract["r"]
                            print("抽象层现在[s, v]:" + str([s, v]))
                            for r in graph_last["relationship"]:
                                print("关系里现在是")
                                print(r)
                                if r["r"] == [s, v] and node_now["text"] != r["source"]["text"]:
                                    rela1 = {"source": r["source"],
                                            "target": node_now,
                                            "r": r["target"]}
                                    graph_new.append(rela1)
    print(graph_new)
    return graph_new


if __name__ == '__main__':
    file_name = "data/abstract.txt"
    # 抽象层储存在txt，不是这次用户上传
    graph_last = {
        'node': [{'name': '阿里健康', 'type': 'P-I', 'ntype': 'actor'}, {'name': '软件服务框架协议', 'type': 'RUV', 'ntype': 'value'},
                 {'name': '淘宝集团', 'type': 'Broker', 'ntype': 'actor'}, {'name': '服务费', 'type': 'ECV', 'ntype': 'value'},
                 {'name': '亚马逊', 'type': 'Broker', 'ntype': 'actor'}, {'name': '医疗保险公司', 'type': 'RUV', 'ntype': 'value'},
                 {'name': '苹果', 'type': 'Provider', 'ntype': 'actor'}, {'name': '供应商报告', 'type': 'INV', 'ntype': 'value'}],
        'rela': [{'source': '阿里健康', 'target': '软件服务框架协议', 'r': 'A-V'}, {'source': '淘宝集团', 'target': '服务费', 'r': 'A-V'},
                 {'source': '亚马逊', 'target': '医疗保险公司', 'r': 'A-V'}, {'source': '苹果', 'target': '供应商报告', 'r': 'A-V'}]}

    graph = gene_value(graph_last, file_name)
    print(graph)

#遍历实例节点 按种类，抽象层-每一类期望哪几类s-v-t 找到s-v，实例-遍历谁出现实例s-v，出现则和这一类的实例节点连接
# 新的graph中加入连好的，判断图中重复，输出