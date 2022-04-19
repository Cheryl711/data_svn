# encoding:utf-8
from bs4 import BeautifulSoup
# from readability.readability import Document
from lxml import html
from lxml.html.clean import Cleaner
from lxml import etree
# from lxml.etree import HTMLParser
import html as ht
import sys, os, string
from html.parser import HTMLParser
import json


import os.path
import sys
import codecs
import re



def extr_clean_token():
    mapped_f = open("全国小米之家_token_final.json", 'w', encoding='utf-8')
    for line in open("全国_node_token1.json", 'r', encoding='utf-8', errors='ignore').readlines():
        value = json.loads(line)
        if (value["value"] == "" or "：" in value["value"] or "©" in value["value"] or "\\" in value["value"]
                or "/" in value["value"] or ":" in value["value"] or "n" in value["value"] or "  " in value["value"]):
            print("no")
        else:
            mapped_f.write(json.dumps(value, ensure_ascii=False) + '\n')


def read_path_file():
    '''
    读取此目录下 改path
    :param
    '''
    # reload(sys)
    # sys.setdefaultencoding('utf-8')
    file_name_list = []
    path = 'C:/Users/62645/PycharmProjects/data_clean/svn_auto_generate/label_html'
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            file_name_list.append(f)
    return file_name_list
    # 其中第一个为起始路径，第二个为起始路径下的文件夹, 第三个是起始路径下的文件.
    # dirpath是一个string，代表目录的路径,
    # dirnames是一个list，包含了dirpath下所有子目录的名字,
    # filenames是一个list，包含了非目录文件的名字.这些名字不包含路径信息, 如果需要得到全路径, 需要使用
    # os.path.join(dirpath, name)
    # print(root,dirs,files)
    # for dir in dirs:
    #     print(os.path.join(root, dir).decode('gbk').encode('utf-8'))

def read_clean(line):
    '''
    标数据用
    :param output_file： 输出文件
    '''
    line1 = line["value"]
    lines_list = []
    line2 = re.sub('\s+', '', line1)
    line_fields = line2.strip().split('：')
    if len(line_fields) == 2:
        lines = line_fields[1]
    else:
        lines = line_fields[0]
    lines = clean(lines)
    word = lines.split('；',)
    for word_list in word:
        word_split = word_list.split("，")
        for w in word_split:
            word_clean = clean(w)
            lines_list.append(word_clean)
    actor = line["actor"]
    return actor, lines_list


def read_txt():
    '''
    标数据用
    :param output_file： 输出文件
    '''
    file_write =  open("data/labeled.txt", 'w', encoding='utf-8') #输出文件
    df = open("data/data.txt", 'r', encoding='utf8')
    lines = []
    for line in df:
        line = re.sub('\s+', '', line)
        line_fields = line.strip().split('：')
        if len(line_fields)==2:
            lines = line_fields[1]
        else:
            lines = line_fields[0]
        lines = clean(lines)
        word = lines.split('；',)
        for word_list in word:
            word_split = word_list.split("，")
            for w in word_split:
                word_clean = clean(w)
                file_write.write("0" + "\t" + word_clean + '\n')
# 并
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

def clean(sen):
    word=sen.replace("等。", "").replace("。", "").replace("等。", "")
    word1 = re.sub(u"\\(.*?\\)|\\【.*?】|\\[.*?]|\\（.*?）", "", word)
    return word1



def label_json():
    mapped_f = open("./json_data/html_labeled0428.json", 'w', encoding='utf-8')
    #输出文件
    for line in open("./json_data/token0421.json", 'r', encoding='utf-8').readlines():
        #输入文件 、先要改的文件
        value = json.loads(line)
        mapped_f.write("0"+"\t" + json.dumps(value, ensure_ascii=False) + '\n')


def label_change():
    dict_y = {'N': 0,
              'P-S': 1,  # '供应商',
              'PE': 2,  # '营销平台',
              'CE-S': 3,  # '线下售卖门店/体验门店',
              'CE-A': 4,  # '售后服务商',
              'PE-P': 5,  # '摄影拍摄/短视频制作/店铺装修',
              }
    mapped_f = open("./json_data/html_labeled0421.json", 'w', encoding='utf-8')
    for line in open("./json_data/html_labeled2203.json", 'r', encoding='utf-8').readlines():
        line_label = line.split('\t')
        label = line_label[0]
        line_value = line_label[1:]
        line_value = line_value[0].replace("\n", "")
        labeled_value = json.loads(line_value)
        int_label = dict_y[label]
        mapped_f.write(str(int_label) + "\t" + json.dumps(labeled_value, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # file_name = './data/小米线下专卖店 - 小米商城.html'
    # input_process_html(file_name)

    read_txt()



