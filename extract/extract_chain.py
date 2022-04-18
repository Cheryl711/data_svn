import json
import copy
import embedding
import os


def extract_cbp(source, target):
    f = open(source, "r", encoding="utf-8")
    output = open(target, "w", encoding="utf-8")
    data = json.load(f)
    out = {}

    out["orphans"] = data["orphans"]
    out["diagram"] = data["diagram"]
    out["display"] = {}

    out["actors"] = []
    out["links"] = []
    out["dependencies"] = []

    rel_list = []
    customer_list = []
    # 认为没有重复的cb关系， provider也只有自己的一条链
    provider_list = []
    cb_rel_list = []

    for d in data["dependencies"]:
        text = d["text"]
        rel = d
        s_id = d["source"]
        t_id = d["target"]
        for n in data["actors"]:
            if n["id"] == s_id:
                rel["source"] = n
            if n["id"] == t_id:
                rel["target"] = n
        rel_list.append(rel)
        # 形成v-source-target
    for ac in data["actors"]:
        if ac["text"].startswith("顾客"):
            customer_list.append(ac["id"])

    # 遍历顾客，针对每个顾客；找到c-b关系储存下来并判断;n是c-b的关系
    for id in customer_list:
        broker_list = []
        rela_list = []
        cb_text_list = []
        # rela_list所有c-b/b-c的关系
        for n in rel_list:
            if n["source"]["id"] == id and n["target"]["ActorType"] == "Broker":
                rela_list.append(n)
                if n["target"]["id"] not in broker_list:
                    broker_list.append(n["target"]["id"])
            if n["target"]["id"] == id and n["source"]["ActorType"] == "Broker":
                rela_list.append(n)
                if n["source"]["id"] not in broker_list:
                    broker_list.append(n["source"]["id"])

        for b in broker_list:
            # 遍历broker，针对每个broker；找到b-p关系储存下来并判断;
            brela_list = []
            value_list = []
            for n in rel_list:
                if n["source"]["id"] == b and n["target"]["ActorType"] == 'Provider':
                    # n是b - p
                    provider = n["target"]
                    for nn in rel_list:
                        if nn["target"]["id"] == b and nn["source"]["id"] == provider["id"]:
                            # 模式一：c-b-p 单条链返回 n是b-p nn是p-b l是c-b/b-c
                            y = provider["y"]
                            n1 = copy.deepcopy(n)
                            n1["source"]["id"] = n["source"]["id"][:18] + n["target"]["id"][18:]
                            n1["source"]["y"] = y
                            nn1 = copy.deepcopy(nn)
                            nn1["target"] = copy.deepcopy(n1["source"])
                            # 在改n1["source"] 是broker； nn的target是broker
                            for l in rela_list:
                                if l["text"] == n["text"] and l["id"] != n["id"]:
                                    # 在改l是c-b； ["source"] 是customer； target是broker
                                    l1 = copy.deepcopy(l)
                                    l1["source"]["id"] = l["source"]["id"][:18] + provider["id"][18:]
                                    l1["source"]["y"] = y
                                    l1["target"] = copy.deepcopy(n1["source"])
                                    cb_rel_list.append(l1["text"])
                                    provider_list.append(provider["text"])
                                    brela_list.append(l1)
                                    brela_list.append(n1)  # 在改n1["source"] 是broker； nn的target是broker
                                    brela_list.append(nn1)
                                elif l["text"] == nn1["text"] and l["id"] != nn1["id"]:
                                    l1 = copy.deepcopy(l)
                                    l1["target"]["id"] = l["target"]["id"][:18] + provider["id"][18:]
                                    l1["target"]["y"] = y
                                    l1["source"] = copy.deepcopy(n1["source"])
                                    cb_rel_list.append(l1["text"])
                                    provider_list.append(provider["text"])
                                    brela_list.append(l1)
                                    if (n1 not in brela_list) and (nn1 not in brela_list):
                                        # 在改n1["source"] 是broker； nn的target是broker
                                        brela_list.append(n1)
                                        brela_list.append(nn1)
                        elif nn["target"]["ActorType"] == 'Customer' and nn["source"]["id"] == provider["id"]:
                            # 模式二： c-b-p 返回p-c n是b-p nn是p-c l是c-b
                            y = provider["y"]
                            n1 = copy.deepcopy(n)
                            nn1 = copy.deepcopy(nn)
                            # 新建n中的b
                            n1["source"]["id"] = n1["source"]["id"][:19] + n1["target"]["id"][19:]
                            n1["source"]["y"] = y
                            nn1["target"]["id"] = nn1["target"]["id"][:19] + nn1["source"]["id"][19:]
                            nn1["target"]["y"] = y
                            nn1["y"] = nn1["y"] + 10
                            nn1["x"] = nn1["x"] - 40
                            # 新建nn中的c
                            for l in rela_list:
                                if l["text"] == n["text"] and l["target"]["ActorType"] == 'Broker' \
                                        and l["source"]["ActorType"] == 'Customer':
                                    # 根据n的b-p价值去找 l的 c-b价值
                                    # 找到后 根据已有的nn中的c去改l的source 和 n1已有的b去改l的target
                                    l1 = copy.deepcopy(l)
                                    l1["source"] = copy.deepcopy(nn1["target"])
                                    l1["target"] = copy.deepcopy(n1["source"])
                                    if l1["id"] not in value_list:
                                        value_list.append(l1["id"])
                                        brela_list.append(l1)
                                    else:
                                        l1["id"] = l1["id"][:18] + provider["id"]
                                        l1["y"] = y
                                        brela_list.append(l1)
                                    # 加入cb和provide唯一的条件
                                    cb_rel_list.append(l1["text"])
                                    provider_list.append(provider["text"])
                                    brela_list.append(nn1)
                                    brela_list.append(n1)
            for cb in rela_list:
                value_list = []
                if cb["text"] not in cb_rel_list and cb["target"]["ActorType"] == "Broker":
                    # 模式三：c-ce-b-p 复合 没有b-p间价值能和c-b间价值对应，如果存在c-b间总价值
                    # cb, c-ce, ce-b, n是b-p nn是p-b  可能有多个 # 遍历cb，找对应b-p再找p-b， 再找b-ce/ce-c
                    # 其中p-b对应 -- b-ce+ce-c 或者 p-b对应 b-c
                    for n in rel_list:
                        if n["source"]["id"] == b and n["target"]["ActorType"] == 'Provider' \
                                and n["target"]["text"] not in provider_list:
                            # n是b - p
                            provider = copy.deepcopy(n["target"])
                            y = provider["y"]
                            n1 = copy.deepcopy(n)
                            #  n1["source"] -- broker
                            n1["source"]["id"] = n1["source"]["id"][:18] + provider["id"][18:] + cb["source"]["id"][:18]
                            n1["source"]["y"] = y

                            cb1 = copy.deepcopy(cb)
                            #  cb1["source"] -- customer
                            cb1["id"] = cb1["id"] + provider["id"][10:]
                            cb1["source"]["id"] = cb1["source"]["id"][:18] + provider["id"][10:]
                            cb1["source"]["y"] = y
                            cb1["target"] = copy.deepcopy(n1["source"])
                            cb_text_list.append(cb1["text"])

                            if is_same_value(cb1["text"], n1["text"]):
                                # 判断b-p对应c-b的分解价值，provider确定了
                                for nnn in rel_list:
                                    if nnn["target"]["id"] == b and nnn["source"]["id"] == provider["id"]:
                                        # nn 是p-b
                                        # print(nnn["text"])
                                        nn1 = copy.deepcopy(nnn)
                                        nn1["target"] = copy.deepcopy(n1["source"])
                                        p_b = nn1["text"]
                                        for nn in rel_list:
                                            if nn["source"]["id"] == b and nn["target"]["ActorType"] == 'C-enabler' and \
                                                    nn["text"] == p_b:
                                                # 其中p-b对应 -- b-ce+ce-c
                                                # b-ce, ce-c 认为ce唯一
                                                ce_list = []
                                                b_ce = copy.deepcopy(nn)
                                                # b-ce 改source-broker
                                                b_ce["source"] = copy.deepcopy(n1["source"])
                                                # b-ce 新建ce
                                                b_ce["target"]["id"] = cb1["source"]["id"][:18] + b_ce["id"][10:]
                                                ce_id = b_ce["target"]["id"]
                                                ce_list.append(b_ce["text"])

                                                brela_list.append(cb1)
                                                brela_list.append(n1)
                                                brela_list.append(nn1)
                                                brela_list.append(b_ce)
                                                for l in rel_list:
                                                    if l["text"] not in ce_list and l["source"]["ActorType"] == 'Broker' \
                                                            and l["target"]["id"] == ce_id:
                                                        # l是b-ce； ["source"] 是broker（改）； target是ce
                                                        b_ce1 = copy.deepcopy(l)
                                                        # b-ce 改source-broker
                                                        b_ce1["source"] = copy.deepcopy(n1["source"])

                                                        ce_list.append(b_ce1["text"])
                                                        brela_list.append(b_ce1)
                                                    elif l["text"] not in ce_list and l["target"][
                                                        "ActorType"] == 'Customer' \
                                                            and l["source"]["id"] == ce_id:
                                                        # l是ce-c； target是c(改)
                                                        ce_c = copy.deepcopy(l)
                                                        ce_c["target"] = copy.deepcopy(cb1["source"])

                                                        ce_list.append(ce_c["text"])
                                                        brela_list.append(ce_c)

                                            elif nn["source"]["id"] == b and nn["target"]["ActorType"] == 'Customer' and \
                                                    nn["text"] == p_b:
                                                # p-b对应（复合-分解） b-c 还有正常的b-ce+ce-c
                                                # b-c
                                                ce_list = []
                                                b_c = copy.deepcopy(nn)
                                                b_c["source"] = copy.deepcopy(n1["source"])
                                                b_c["target"] = copy.deepcopy(cb1["source"])
                                                ce_list.append(b_ce["text"])

                                                brela_list.append(cb1)
                                                brela_list.append(n1)
                                                brela_list.append(nn1)
                                                brela_list.append(b_c)
                                                for l in rel_list:
                                                    if l["target"]["ActorType"] == 'C-enabler' and \
                                                            l["source"]["ActorType"] == 'Broker' \
                                                            and is_same_value(cb1["text"], n1["text"]):
                                                        # l是b-ce； ["source"] 是broker（改）； target是ce
                                                        b_ce = copy.deepcopy(l)
                                                        # b-ce 改source-broker
                                                        b_ce["source"] = copy.deepcopy(n1["source"])
                                                        brela_list.append(b_ce)
                                                        ce = b_ce["target"]["id"]
                                                        for ce_c in rel_list:
                                                            if ce_c["source"]["id"] == ce:
                                                                # l是ce-c； target是c(改)
                                                                ce_c1 = copy.deepcopy(ce_c)
                                                                ce_c1["target"] = copy.deepcopy(cb1["source"])
                                                                brela_list.append(ce_c1)
            # print(brela_list)
            actor_list = []
            for i in brela_list:
                link1 = {
                    "id": "",
                    "type": "istar.DependencyLink",
                    "source": "",
                    "target": ""
                }
                link2 = {
                    "id": "",
                    "type": "istar.DependencyLink",
                    "source": "",
                    "target": ""
                }
                dependencies1 = {
                    "id": "0d3d1e38-17ea-47a3-b845-bbe4da597d86",
                    "text": "资源使用类价值",
                    "type": "istar.Value",
                    "x": 272,
                    "y": 1045,
                    "customProperties": {
                    },
                    "source": "42f54568-6aca-431b-aa91-a0bc4893d83e",
                    "target": "aaa1ac30-9ae6-497e-bcae-31e8aa3371ee"
                }
                dependencies1 = copy.deepcopy(i)
                dependencies1["source"] = i["source"]["id"]
                dependencies1["target"] = i["target"]["id"]
                out["dependencies"].append(dependencies1)

                link1["source"] = i["source"]["id"]
                link1["target"] = i["id"]
                link1["id"] = i["source"]["id"][:18] + i["id"]
                link2["source"] = i["id"]
                link2["target"] = i["target"]["id"]
                link2["id"] = i["id"] + i["target"]["id"][18:]
                out["links"].append(link1)
                out["links"].append(link2)
                if i["source"]["id"] not in actor_list:
                    actor_list.append(i["source"]["id"])
                    out["actors"].append(i["source"])
                if i["target"]["id"] not in actor_list:
                    actor_list.append(i["target"]["id"])
                    out["actors"].append(i["target"])

    for actor in out["actors"]:
        if actor["id"] not in out["display"].values():
            id = actor["id"]
            color = "#CCFACD"
            if actor["ActorType"] == "Broker":
                color = "#FAE573"
            elif actor["ActorType"] == "Provider":
                color = "#A0CCFA"
            elif actor["ActorType"] == "C-enabler":
                color = "#CD98FA"
            elif actor["ActorType"] == "P-enabler":
                color = "#FA9457"
            out["display"][id] = {
                "collapsed": "true",
                "backgroundColor": color
            }
    output.write(json.dumps(out, ensure_ascii=False, indent=4) + '\n')


def is_same_value(a, b):
    a = a.replace("费用", "").replace("产品使用权", "")
    b = b.replace("费用", "").replace("产品使用权", "")
    a = a.split("：")[1]
    b = b.split("：")[1]
    corr = embedding.count_corr(a, b)
    if corr > 0.6:
        return True
    else:
        return False


if __name__ == '__main__':
    file_path = 'data/IVN/'
    for file_name in os.listdir(file_path):
        source = file_path + file_name
        target = 'data/DVC/' + file_name
        print('\n' + '-' * 20 + '开始抽取' + file_name + '文件' + '-' * 20)
        extract_cbp(source, target)
        print('-' * 25 + '抽取完成' + '-' * 25 + '\n')
