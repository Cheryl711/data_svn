from flask import Flask, request, render_template
from flask import url_for, request, session, flash, redirect
# from wtforms import MultipleFileField
from flask import jsonify
from flask_cors import CORS
from ast_html.classification import *
from extraction.mapping import *
import classification.classify
import classification.generate_abs


ERROR = 'error'
SUCC = 'succ'

app = Flask(__name__, static_folder='./ast_html/html', template_folder='./web/dist')

@app.route('/api/add-review', methods=['POST'])
def app_add_review():
    f = request.files['file']
    print(f.read(0))
    return SUCC


@app.route('/api/hello')
def app_hello():
    return SUCC


@app.route('/api/multi-upload', methods=["POST"])
def multi_upload1():
    uploaded_files = request.files.getlist("file")
    print(uploaded_files)
    broker = request.values.get("broker")
    print(broker)
    return jsonify(html_classify(broker, uploaded_files))


@app.route('/api/classify-whole', methods=["POST"])
def classify_whole22():
    f = request.files['file']
    broker = request.values.get("node")
    return jsonify(classify_whole(broker, f))


@app.route('/api/classify-value1', methods=["POST"])
def classify_whole121():
    f = request.files['file']
    broker = request.values.get("node")
    return jsonify(classify_single(broker, f))


@app.route('/api/classify-value2', methods=["POST"])
def classify_whole122():
    f = request.files['file']
    broker = request.values.get("node")
    return jsonify(classification.classify.get_scope_classify(f, broker))


@app.route('/api/generate', methods=["POST"])
def generate13():
    # f = request.files['file']
    broker = request.values.get("node")
    return jsonify(classification.generate_abs.find_rel(broker))


if __name__ == '__main__':
    CORS(app, supports_credentials=True)
    app.run(host='0.0.0.0', port=45006, debug=True)

