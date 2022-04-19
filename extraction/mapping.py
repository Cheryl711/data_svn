import re
import json
# import sys
# sys.path.append("..")
import extraction.joint_learning
import extraction.classify.classify_object

def is_multi_sentence(line):
    '''
    This function is used to judge whether a line contains multi line.
    @return boolean
    '''
    return len(line.split(',')) > 3


def extra_labels_and_words(line, ignore_lables=['attribute']):
    '''
    @description: this function is used to extra lables and corresponding words
        in a line
    @paramters: line --> str
    @paramters: ignore_lables --> list of labels needed to be ignore
    @return: labels--> list ;  words --> list
    '''
    pattern = re.compile(r'{{([^:]+):([^:]+)}}')
    labels, words = [], []
    for m in pattern.finditer(line):
        if m.group(1) in ignore_lables:
            continue
        labels.append(m.group(1))
        words.append(m.group(2))
    return labels, words

def pattern_mapping(line_r):
    labels_r, words_r = extra_labels_and_words(line_r)
    #Object>1, Actor/recipient>2
    num_object = 0
    num_actor = 0
    num_recipient = 0
    edge={}
    for label in labels_r:
        if label == "object":
            num_object = num_object + 1
        if label == "actor" :
            num_actor = num_actor + 1
        if label == "recipient":
            num_recipient = num_recipient + 1
    if num_actor >= 1 and num_object >= 1 and num_recipient>= 1:
        match_actor_pattern = "actor"
        matched_actor_index = labels_r.index(match_actor_pattern)
        match_recipient_pattern = "recipient"
        matched_recipient_index = labels_r.index(match_recipient_pattern)
        match_object_pattern = "object"
        matched_object_index = labels_r.index(match_object_pattern)
        location = 0
        for i in range(labels_r.count(match_object_pattern)):
            location += labels_r[location:].index(match_object_pattern)
            # print(location)
            location += 1
        edge = {"source": words_r[matched_actor_index],
                "target": words_r[matched_recipient_index],
                "r": words_r[matched_object_index]}
        print(edge)
        return edge

def pattern_mapping_single(line_r):
    labels_r, words_r = extra_labels_and_words(line_r)
    #Object>1, Actor/recipient>1
    num_object = 0
    num_recipient = 0
    num_actor = 0
    edge={}
    for label in labels_r:
        if label == "object":
            num_object = num_object + 1
        if label == "recipient":
            num_recipient = num_recipient + 1
        if label == "actor":
            num_actor = num_actor + 1
    if num_actor >= 1 and num_recipient == 0 and num_object >= 1:
        match_actor_pattern = "actor"
        matched_actor_index = labels_r.index(match_actor_pattern)
        match_object_pattern = "object"
        matched_object_index = labels_r.index(match_object_pattern)
        location = 0
        for i in range(labels_r.count(match_object_pattern)):
            location += labels_r[location:].index(match_object_pattern)
            location += 1
        edge = {"source": words_r[matched_actor_index],
                "r": words_r[matched_object_index]}
        return edge

def extra_structure_information(line):
    '''
    NOTE: if relation_type is `wrong`, this sample need to be reviewed by human.
    '''
    relation_type = 'single'
    if is_multi_sentence(line):
        relation_type, line_l, line_r, time = tuple(line.split(','))
        if relation_type == 'single':
            relation_type = 'wrong'
        labels_l, words_l = extra_labels_and_words(line_l)
        labels_r, words_r = extra_labels_and_words(line_r)
        labels, words = [labels_l, labels_r], [words_l, words_r]
    else:
        relation_type, line, time = tuple(line.split(','))
        if relation_type != 'single':
            relation_type = 'wrong'
        labels, words = extra_labels_and_words(line)
    return relation_type, labels, words, time


def is_equal(str1, list2):
    i = False
    str2 = ""
    for str2 in list2:
        if str1 == str2 or str1 in str2 or str2 in str1:
            if len(str1) == len(str2):
                i = True
                break
            if len(str1) >= len(str2):
                l1 = len(str1)
                up = len(str2) + 2
                down =  len(str2) - 2
                if l1>=down and l1 <=up :
                    i = True
                    break
            if len(str1) <= len(str2):
                l1 = len(str2)
                up = len(str1) + 2
                down =len(str1) - 2
                if l1>=down and l1 <=up :
                    i = True
                    break
    return i, str2

def is_equal_str(str1, str2):
    i = False
    if str1 == str2 or str1 in str2 or str2 in str1:
        if len(str1) == len(str2):
            i = True
        if len(str1) >= len(str2):
            l1 = len(str1)
            up = len(str2) + 2
            down =  len(str2) - 2
            if l1>=down and l1 <=up :
                i = True
        if len(str1) <= len(str2):
            l1 = len(str2)
            up = len(str1) + 2
            down =len(str1) - 2
            if l1>=down and l1 <=up :
                i = True
    return i

def gene_graph(node_g, value, edge_list):
    n_list = []
    graph = {
    "actors" :[],
    "rela" :[]
    }
    graph_name_list = []
    node_g = json.loads(node_g)
    if node_g is None or node_g["actors"] == []:
        for object, edg in zip(value, edge_list):
            source = edg["source"]
            target = edg["target"]
            if int(object["label"]) == 1:
                node1 = {
                    "text": source,
                    "ActorType": "Provider",
                    "type": "istar.Actor"
                }
                i, source_is = is_equal(source, graph_name_list)
                if i:
                    source = source_is
                    node1["text"] = source
                else:
                    graph["actors"].append(node1)
                    graph_name_list.append(source)

                node2 = {
                    "text": target,
                    "ActorType": "Provider",
                    "type": "istar.Actor"
                }
                i, target_is = is_equal(target, graph_name_list)
                if i:
                    target = target_is
                else:
                    graph["actors"].append(node2)
                    graph_name_list.append(target)

                node3 = {
                    "text": edg["r"],
                    "Type": "RUV",
                    "type": "istar.Value"
                }
                if node3 not in graph["actors"]:
                    graph["actors"].append(node3)

                rel1 = {
                    "source": node1,
                    "target": node2,
                    "r": node3,
                }
                if rel1 not in graph["rela"]:
                    graph["rela"].append(rel1)
    else:
        for n in node_g["actors"]:
            n_list.append(n["text"])
        for object, edg in zip(value, edge_list):
            source = edg["source"]
            target = edg["target"]
            has_source = False
            has_target = False
            if int(object["label"]) == 1:
                node1 = {
                    "text": source,
                    "ActorType": "Provider",
                    "type": "istar.Actor"
                }
                i, source_is = is_equal(source, graph_name_list)
                if i:
                    source = source_is
                    node1["text"] = source
                    has_source = True
                else:
                    for node in node_g["actors"]:
                        if is_equal_str(source, node["text"]):
                            node1["ActorType"] = node["ActorType"]
                            graph_name_list.append(node["text"])
                            source = node["text"]
                            node1["text"] = source
                            has_source = True
                            break
                    if has_source:
                        graph["actors"].append(node1)
                        graph_name_list.append(source)

                node2 = {
                    "text": target,
                    "ActorType": "Provider",
                    "type": "istar.Actor"
                }
                i, target_is = is_equal(target, graph_name_list)
                if i:
                    target = target_is
                    node2["text"] = target
                    has_target = True
                else:
                    for node in node_g["actors"]:
                        if is_equal_str(target, node["text"]):
                            node2["ActorType"] = node["ActorType"]
                            graph_name_list.append(node["text"])
                            target = node["text"]
                            node2["text"] = target
                            has_target = True
                            break
                        if has_target:
                            graph["actors"].append(node2)
                            graph_name_list.append(target)

                if has_source or has_target:
                    if has_target is False:
                        graph["actors"].append(node2)
                        graph_name_list.append(target)
                    if has_source is False:
                        graph["actors"].append(node1)
                        graph_name_list.append(source)
                    node3 = {
                        "text": edg["r"],
                        "Type": "RUV",
                        "type": "istar.Value"
                    }
                    if node3 not in graph["actors"]:
                        graph["actors"].append(node3)
                    rel1 = {
                        "source": node1,
                        "target": node2,
                        "r": node3,
                    }
                    if rel1 not in graph["rela"]:
                        graph["rela"].append(rel1)
    return graph


def gene_graph_single(node_g, value, edge_list):
    n_list = []
    graph = {
        "actors": [],
        "rela": [],
        "nodes": []
        # nodes是没有关系的单独节点
             }
    graph_name_list = []
    node_g = json.loads(node_g)
    if node_g is None or node_g["actors"] == []:
        for object, edg in zip(value, edge_list):
            source = edg["source"]
            if int(object["label"]) == 1:
                node1 = {
                    "text": source,
                    "ActorType": "Provider",
                    "type": "istar.Actor"
                }
                i, source_is = is_equal(source, graph_name_list)
                if i:
                    source = source_is
                    node1["text"] = source
                else:
                    graph["actors"].append(node1)
                    graph_name_list.append(source)

                node3 = {
                    "text": edg["r"],
                    "Type": "RUV",
                    "type": "istar.Value"
                }
                if node3 not in graph["actors"]:
                    graph["actors"].append(node3)

                rel1 = {
                    "source": node1,
                    "r": node3,
                }
                if rel1 not in graph["rela"]:
                    graph["rela"].append(rel1)
    else:
        for object, edg in zip(value, edge_list):
            source = edg["source"]
            has_source = False
            if int(object["label"]) == 1:
                node1 = {
                    # "text": source,
                    # "ActorType": "Provider",
                    "type": "istar.Actor",
                    # "Atype":
                }
                i, source_is = is_equal(source, graph_name_list)
                if i:
                    for actor in graph["actors"]:
                        if actor["text"] == source_is:
                            node1 = actor
                    has_source = True
                else:
                    for node in node_g["actors"]:
                        if is_equal_str(source, node["text"]):
                            node1["ActorType"] = node["ActorType"]
                            node1["Atype"] = node["customProperties"]["Atype"]
                            graph_name_list.append(node["text"])
                            source = node["text"]
                            node1["text"] = source
                            has_source = True
                            break
                    if has_source:
                        graph["actors"].append(node1)
                        graph_name_list.append(source)

                if has_source:
                    node3 = {
                        "text": edg["r"],
                        "Type": "RUV",
                        "type": "istar.Value"
                    }
                    if node3 not in graph["actors"]:
                        graph["actors"].append(node3)
                    rel1 = {
                        "source": node1,
                        "r": node3,
                    }
                    if rel1 not in graph["rela"]:
                        graph["rela"].append(rel1)
        for actor in node_g["actors"]:
            if actor["text"] not in graph_name_list:
                node_not_in = {
                    # "text": source,
                    # "ActorType": "Provider",
                    "type": "istar.Actor",
                    # "Atype":
                }
                node_not_in["ActorType"] = actor["ActorType"]
                node_not_in["Atype"] = actor["customProperties"]["Atype"]
                node_not_in["text"] = actor["text"]
                graph["nodes"].append(node_not_in)
    return graph


def classify_whole_experiment(node_g, file_name):
    '''
    方法二 step2得到完整价值网
    :param node: step1矫正后节点格式：{'node': [{'name': '阿里健康', 'type': 'CE', 'ntype': 'actor'},
                                                {'name': '淘宝', 'type': 'B', 'ntype': 'actor'}])
    :param file: 上传文件名称，根据接口待修改
    :return graph: 得到的节点和边
    '''
    edge_list = []
    value_list = []
    graph = {
        "node" :[],
        "rela" :[]
    }
    sen = extraction.joint_learning.p_run(file_name)
    for line in sen:
        edge = pattern_mapping(line)
        if edge is not None:
            edge_list.append(edge)
            value_list.append(edge["r"])
    value = extraction.classify.classify_object.predict_data(value_list)
    # 判断edge["r"]中value是不是价值，返回{'label': '1', 'object': '软件服务框架协议'},
    print(edge_list)
    print (value)
    for object, edg in zip(value, edge_list):
        if int(object["label"]) == 1:
            print(edg)
    # graph = gene_graph(node_g, value, edge_list)
    # print(graph)
    # return graph
    return sen

def classify_single_experiment(sen, node_g, file_name):
    '''
    方法一 step2文件格式一
    :param node: step1矫正后节点格式：{'node': [{'name': '阿里健康', 'type': 'CE', 'ntype': 'actor'},
                                                {'name': '淘宝', 'type': 'B', 'ntype': 'actor'}])
    :param file: 上传文件名称，根据接口待修改
    :return graph: 得到的节点和边
    '''
    edge_list = []
    value_list = []
    # sen = extraction.joint_learning.p_run(file_name)
    for line in sen:
        edge = pattern_mapping_single(line)
        if edge is not None:
            edge_list.append(edge)
            value_list.append(edge["r"])
    value = extraction.classify.classify_object.predict_data(value_list)
    print(value)
    print(edge_list)
    for object, edg in zip(value, edge_list):
        if int(object["label"]) == 1:
            print(edg)


def classify_whole(node_g, file_name):
    '''
    方法二 step2得到完整价值网
    :param node: step1矫正后节点格式：{'node': [{'name': '阿里健康', 'type': 'CE', 'ntype': 'actor'},
                                                {'name': '淘宝', 'type': 'B', 'ntype': 'actor'}])
    :param file: 上传文件名称，根据接口待修改
    :return graph: 得到的节点和边
    '''
    edge_list = []
    value_list = []
    graph = {
        "node" :[],
        "rela" :[]
    }
    sen = extraction.joint_learning.p_run(file_name)
    for line in sen:
        edge = pattern_mapping(line)
        if edge is not None:
            edge_list.append(edge)
            value_list.append(edge["r"])
    value = extraction.classify.classify_object.predict_data(value_list)
    # 判断edge["r"]中value是不是价值，返回{'label': '1', 'object': '软件服务框架协议'},
    graph = gene_graph(node_g, value, edge_list)
    print(graph)
    return graph


def classify_single(node_g, file_name):
    '''
    方法一 step2文件格式一
    :param node: step1矫正后节点格式：{'node': [{'name': '阿里健康', 'type': 'CE', 'ntype': 'actor'},
                                                {'name': '淘宝', 'type': 'B', 'ntype': 'actor'}])
    :param file: 上传文件名称，根据接口待修改
    :return graph: 得到的节点和边
    '''
    edge_list = []
    value_list = []
    graph = {
    "node" :[],
    "rela" :[]
    }
    sen = extraction.joint_learning.p_run(file_name)
    for line in sen:
        edge = pattern_mapping_single(line)
        if edge is not None:
            edge_list.append(edge)
            value_list.append(edge["r"])
    value = extraction.classify.classify_object.predict_data(value_list)
    print(value)
    print(edge_list)
    graph = gene_graph_single(node_g, value, edge_list)
    return graph


if __name__ == '__main__':
    # raw_filepath = '/home/lmy/git_project/eskm/data/test.txt'
    # label_filepath = '/home/lmy/git_project/eskm/data/label_test.txt'
    # rel_filepath = '/home/lmy/git_project/eskm/data/rel_test.txt'
    # output_filepath = '/home/lmy/git_project/event_mapping/data/history/history_encoded.txt'
    # encode_files(raw_filepath, label_filepath, rel_filepath, output_filepath)
    node = []
    file_name = "data/test.txt"
    # node = [{'name': '阿里健康', 'type': 'CE', 'ntype': 'actor'},
    #         {'name': '淘宝', 'type': 'B', 'ntype': 'actor'}]
    # graph = classify_whole(node, file_name)
    sen = classify_whole_experiment(node, file_name)
    classify_single_experiment(sen, node, file_name)




    # raw_filepath= 'test.txt'
    # raw_lines = open(raw_filepath, 'r', encoding='utf8').readlines()
    # for line in raw_lines:
    #     pattern_mapping(line)


