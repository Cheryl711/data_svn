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


def pre_readability():
    f = open('./全国.html', 'r', encoding='utf-8')
    htmlpage = f.read()
    # res = BeautifulSoup(htmlpage, "html.parser")
    res = htmlpage
    # res = requests.get('http://finance.sina.com.cn/roll/2019-02-12/doc-ihrfqzka5034116.shtml')
    # 获取新闻标题
    readable_title = Document(res).short_title()
    # 获取内容并清洗
    readable_article = Document(res).summary()
    print(readable_article)
    # text_p = re.sub(r'</?div.*?>', '', readable_article)
    # text_p = re.sub(r'((</p>)?<a href=.*?>|</a>(<p>)?)', '', text_p)
    # text_p = re.sub(r'<select>.*?</select>', '', text_p)


def clean_script(html_str):
    cleaner = Cleaner()
    cleaner.javascript = True  # This is True because we want to activate the javascript filter
    cleaner.style = True  # clean the style element
    tree = html.fromstring(html_str)
    print(html.tostring(cleaner.clean_html(tree)))


def remove_node(html_str):
    '''
    clean the html
    :param html_str: html to be processed
    :return output：cleaned html
    '''
    # mapped_f = open(input_clean_file, 'w', encoding='utf-8')
    # cleaner = Cleaner(style=True, scripts=True, page_structure=False, safe_attrs_only=False)
    cleaner = Cleaner(meta=False, page_structure=False, safe_attrs_only=False)
    tree = html.fromstring(html_str)
    tree = cleaner.clean_html(tree)
    # tree = etree.HTML(html_str, etree.HTMLParser())
    # ele = tree.xpath('//script')
    # for e in ele:
    #     e.getparent().remove(e)
    Html = html.tostring(tree).decode()
    output = ht.unescape(Html)
    # unescape()将字符串中的uncode变化转成中文
    # mapped_f.write(output)
    return output

    # html = etree.HTML(text)  # 初始化生成一个XPath解析对象
    # result = etree.tostring(html, encoding='utf-8')  # 解析对象输出代码
    # print(type(html))
    # print(type(result))
    # print(result.decode('utf-8'))
    # ele = tree.xpath('//script | //noscript')

    # tree = etree.HTML(html_str.decode('utf-8'))
    # output=html.tostring(tree, encoding='utf-8')
    # mapped_f.write(output.decode('utf-8'))


def get_all_node(html_str):
    mapped_f = open("全国_clean.html", 'w', encoding='utf-8')
    tree = html.fromstring(html_str)
    result = tree.xpath('//*[count(*) eq 0].')  # 获取a节点下的内容
    result1 = tree.xpath('//li[@class="item-1"]//text()')  # 获取li下所有子孙节点的内容
    print(result)
    # Html = html.tostring(tree).decode()
    # output=ht.unescape(Html)
    # mapped_f.write(output)


def get_token(html, output_file):
    '''
    get the param from html
    :param html: html to be processed
    :param output_file
    '''
    # f = open(input_file, 'r', encoding='utf-8')
    # html = f.read()
    title = ""
    keyword = ""
    tagstack = []
    out_f = open(output_file, 'a', encoding='utf-8')
    delete_list = ["！", "：", "©", "\\", "+", ">", "¥", "|", "/", ":", "n", "  ", ".", "[", "-", "英寸", '首页',
                   '入驻流程', '入驻规则', '退出', '上一页', '下一页', 'shadow', '点击可查看详情','登录', '注册', "上一张",
                   "下一张", "共", "销量", "点击查询", "×", "”", "“", "NEW", "和", "。", "、", "广告", "{title}", "",
                   "￥", "◆", "大事记", '上一个', '下一个',]

    class MyHTMLParser(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.title = ""
            self.keyword = ""
            self.context = ""
            self.highlight_label = [ 'i', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            self.tem_data = ""
            self.tem_tag = ""

        def handle_starttag(self, tag, attrs):
            """
            recognize start tag, like <div>
            :param tag:
            :param attrs:
            :return:
            """
            if tag != "img" and tag != "br":
                # tag= tag+str(attrs)
                tagstack.append(tag)

            if tag == "meta":
                is_keyword = False
                for name, value in attrs:
                    if value == 'keywords':
                        is_keyword = True
                    if name == 'content':
                        if is_keyword == True:
                            self.keyword = value
                # output="Encountered a start tag:"+str(tagstack)
                # out_f.write(output+ '\n')

        def handle_endtag(self, tag):
            """
            recognize end tag, like </div>
            :param tag:
            :return:
            """
            tagstack.pop()
            # output="Encountered a end tag:"+str(tag)
            # mapped_f.write(output+ '\n')

        def handle_data(self, data):
            """
            recognize data, html content string
            :param data:
            :return:
            """
            # print("Encountered some data  :", data)
            # 处理 上下文 和 title
            for tag in tagstack:
                final_token = tag
            if final_token == "title":
                self.title = data
            if (data.isdigit() or data == "" or "！" in data or "©" in data or "\\" in data
                    or "+" in data or ">" in data or "¥" in data or "|" in data or "/" in data or ":" in data
                    or "n" in data or "  " in data or "." in data  or "[" in data or  "-" in data  or "英寸" in data
                    or "," in data or "_" in data or "，" in data or "%" in data or "·" in data or "【 " in data
                    or "】" in data or "价格 " in data or "？"in data  or "~"in data or "\t\t" in data or "<" in data
                    or "分 " in data or "#" in data or "“" in data or "${" in data or "{{" in data or "登录" in data
                    or "我的" in data or "#" in data or "“" in data or "${" in data or "{{" in data or "登录" in data
                    or data.endswith("元") or data.endswith("起")  or data.endswith("页") or data.endswith("人")
                    or data.endswith("笔") or data.endswith("会员") or data.endswith("条点评") or data.endswith("人点评")
                    or data in delete_list ):
                pass
            else:
                if data.strip() and data != "|":
                    final_token = ""
                    data = data.replace('\n', '').replace('\r', '').replace('  ', '')
                    for tag in tagstack:
                        if tag in self.highlight_label:
                            self.context = data
                    current_context = self.context
                    if current_context == "":
                        current_context = self.tem_data
                    keyword = self.keyword
                    if data.endswith('公司'):
                        keyword = "公司"
                    if data.endswith('工作室'):
                        keyword = "工作室"
                    if data.endswith('店'):
                        keyword = "店"
                    if data.endswith('酒店'):
                        keyword = "酒店"
                    if data.endswith('航空'):
                        keyword = "航空"
                    if "影像" in data or "摄影" in data or "传媒" in data or "视觉" in data:
                        keyword = "传媒"
                    lines = re.split(r',|:| |：', data)
                    if '服务商' in lines[0]:
                        if len(lines) > 1:
                            # print(lines)
                            data = str(lines[1])
                            current_context = "服务商"
                    if data != "":
                        obj = {
                            'value': data,
                            'context': current_context,
                            'token': tagstack,
                            'title': self.title,
                            'keyword': keyword,
                        }
                        if final_token != "title" and obj["value"] != "" and obj["value"].strip():
                            if (obj["value"].isdigit()or obj["value"] == "" or "！" in obj["value"] or "：" in obj["value"] or "©" in obj["value"]
                                    or "\\" in obj["value"] or "+" in obj["value"] or ">" in obj["value"] or "¥" in obj[
                                        "value"]
                                    or "/" in obj["value"] or ":" in obj["value"] or "n" in obj["value"] or "  " in obj[
                                        "value"]):
                                pass
                            else:
                                out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                                self.tem_data = obj["value"]
                    # print(str(obj))

    parser = MyHTMLParser()
    parser.feed(html)
    out_f.close()


def get_token_list(html):
    '''
    get the param from html
    :param html: html to be processed
    :param output_file
    '''
    html_token_list = []
    # f = open(input_file, 'r', encoding='utf-8')
    # html = f.read()
    title = ""
    keyword = ""
    tagstack = []
    delete_list = ["！", "：", "©", "\\", "+", ">", "¥", "|", "/", ":", "n", "  ", ".", "[", "-", "英寸", '首页',
                   '入驻流程', '入驻规则', '退出', '上一页', '下一页', 'shadow', '点击可查看详情', '登录', '注册', "上一张",
                   "下一张", "共", "销量", "点击查询", "×", "”", "“", "NEW", "和", "。", "、", "广告", "{title}", "",
                   "￥", "◆", "大事记", '上一个', '下一个', ]

    class MyHTMLParser(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.title = ""
            self.keyword = ""
            self.context = ""
            self.highlight_label = ['i', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            self.tem_data = ""
            self.tem_tag = ""

        def handle_starttag(self, tag, attrs):
            """
            recognize start tag, like <div>
            :param tag:
            :param attrs:
            :return:
            """
            if tag != "img" and tag != "br":
                # tag= tag+str(attrs)
                tagstack.append(tag)

            if tag == "meta":
                is_keyword = False
                for name, value in attrs:
                    if value == 'keywords':
                        is_keyword = True
                    if name == 'content':
                        if is_keyword == True:
                            self.keyword = value
                # output="Encountered a start tag:"+str(tagstack)
                # out_f.write(output+ '\n')

        def handle_endtag(self, tag):
            """
            recognize end tag, like </div>
            :param tag:
            :return:
            """
            tagstack.pop()
            # output="Encountered a end tag:"+str(tag)
            # mapped_f.write(output+ '\n')

        def handle_data(self, data):
            """
            recognize data, html content string
            :param data:
            :return:
            """
            # print("Encountered some data  :", data)
            # 处理 上下文 和 title
            for tag in tagstack:
                final_token = tag
            if final_token == "title":
                self.title = data
            if (data.isdigit() or data == "" or "！" in data or "©" in data or "\\" in data
                    or "+" in data or ">" in data or "¥" in data or "|" in data or "/" in data or ":" in data
                    or "n" in data or "  " in data or "." in data or "[" in data or "-" in data or "英寸" in data
                    or "," in data or "_" in data or "，" in data or "%" in data or "·" in data or "【 " in data
                    or "】" in data or "价格 " in data or "？" in data or "~" in data or "\t\t" in data or "<" in data
                    or "分 " in data or "#" in data or "“" in data or "${" in data or "{{" in data or "登录" in data
                    or "我的" in data or "#" in data or "“" in data or "${" in data or "{{" in data or "登录" in data
                    or data.endswith("元") or data.endswith("起") or data.endswith("页") or data.endswith("人")
                    or data.endswith("笔") or data.endswith("会员") or data.endswith("条点评") or data.endswith("人点评")
                    or data in delete_list):
                pass
            else:
                if data.strip() and data != "|":
                    final_token = ""
                    data = data.replace('\n', '').replace('\r', '').replace('  ', '')
                    for tag in tagstack:
                        if tag in self.highlight_label:
                            self.context = data
                    current_context = self.context
                    if current_context == "":
                        current_context = self.tem_data
                    keyword = self.keyword
                    if data.endswith('公司'):
                        keyword = "公司"
                    if data.endswith('工作室'):
                        keyword = "工作室"
                    if data.endswith('店'):
                        keyword = "店"
                    if data.endswith('酒店'):
                        keyword = "酒店"
                    if data.endswith('航空'):
                        keyword = "航空"
                    if "影像" in data or "摄影" in data or "传媒" in data or "视觉" in data:
                        keyword = "传媒"
                    lines = re.split(r',|:| |：', data)
                    if '服务商' in lines[0]:
                        if len(lines) > 1:
                            # print(lines)
                            data = str(lines[1])
                            current_context = "服务商"
                    if data != "":
                        obj = {
                            'value': data,
                            'context': current_context,
                            'token': tagstack,
                            'title': self.title,
                            'keyword': keyword,
                        }
                        if final_token != "title" and obj["value"] != "" and obj["value"].strip():
                            if (obj["value"].isdigit() or obj["value"] == "" or "！" in obj["value"] or "：" in obj[
                                "value"] or "©" in obj["value"]
                                    or "\\" in obj["value"] or "+" in obj["value"] or ">" in obj["value"] or "¥" in obj[
                                        "value"]
                                    or "/" in obj["value"] or ":" in obj["value"] or "n" in obj["value"] or "  " in obj[
                                        "value"]):
                                pass
                            else:
                                html_token_list.append(obj)
                                self.tem_data = obj["value"]
                            # out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    parser = MyHTMLParser()
    parser.feed(html)
    return html_token_list


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


def input_process_html(file_name):
    try:
        f = open(file_name, 'r', encoding='utf-8')
        html_str = f.read()
    except:
        f = open(file_name, 'r', encoding='gbk')
        html_str = f.read()
    print("process")
# # 接口输入文件用：
#     html_str = file_name.read()
    output_html = remove_node(html_str)
    html_token_list = get_token_list(output_html)
    # print(html_token_list)
    return html_token_list


def input_process_label_html(file_name):
    '''
    标数据用
    :param output_file： 输出文件
    '''
    try:
        f = open(file_name, 'r', encoding='utf-8')
        html_str = f.read()
    except:
        f = open(file_name, 'rb')
        html_str = f.read()
    output_html = remove_node(html_str)
    # html_token_list = get_token_list(output_html)
    # print(html_token_list)
    output_file = './json_data/token0421_update.json'
    get_token(output_html, output_file)


def reload_label_json():
    # file_old =  open("./json_data/html_labeled.json", 'r', encoding='utf-8')
    file_write =  open("./json_data/html_labeled_taobaoupdata.json", 'w', encoding='utf-8') #输出文件
    number= 0
    for line in open("./json_data/token0421_update.json", 'r', encoding='utf-8').readlines(): #新生成数据文件
        value = json.loads(line)
        is_write = False
        for line1 in open("./json_data/html_labeled0421.json", 'r', encoding='utf-8').readlines(): #原先文件
            line_label = line1.split('\t')
            label = int(line_label[0])
            line_value = line_label[1:]
            line_value = line_value[0].replace("\n", "")
            labeled_value = json.loads(line_value)
            if labeled_value["value"] == value["value"] and labeled_value["token"] == value["token"]:
                # print(labeled_value["value"])
                if is_write is False:
                    file_write.write(str(label) + "\t" + json.dumps(value, ensure_ascii=False) + '\n')
                    is_write = True
                    number=number+1
        if is_write is False:
            file_write.write("0" + "\t" + json.dumps(value, ensure_ascii=False) + '\n')
    print(number)


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

    reload_label_json()
    # label_change()
    # label_json()
    #
    # file_list = read_path_file()
    # for file_name in file_list:
    #     path = "./label_html/"
    #     file = os.path.join(path,file_name)
    #     print(file)
    #     input_process_label_html(file)

    # f = open('./data/小米线下专卖店 - 小米商城.html', 'r', encoding='utf-8')
    # html_str = f.read()
    # output_html = remove_node(html_str)
    # output_file='./json_data/线下专卖店_token.json'
    # get_token(output_html, output_file)





