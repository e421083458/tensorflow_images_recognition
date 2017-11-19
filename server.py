# -*- coding: utf-8 -*-

from googlenet_dog_model import *
from googlenet_model import *

from tensorflow.python.lib.io import file_io
import pickle
import argparse

import os
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
from flask import Flask, request
import uuid

ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG'])
app = Flask(__name__)
model = googlenet_dog_model()

def load_batch(fpath):
    object = file_io.read_file_to_string(fpath)
    # origin_bytes = bytes(object, encoding='latin1')
    # with open(fpath, 'rb') as f:
    if sys.version_info > (3, 0):
        # Python3
        d = pickle.loads(object, encoding='latin1')
    else:
        # Python2
        d = pickle.loads(object)
    data = d["data"]
    labels = d["labels"]
    return data, labels

def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name

def inference(file_name):
    base_name = os.path.basename(file_name)
    new_url = '/static/%s' % base_name
    image_tag = '<img src="%s" width="100"></img><p>'
    new_tag = image_tag % new_url
    print(new_tag)
    try:
        predict = model.run_inference_on_image(file_name)
    except Exception as ex:
        predict = ex
        print(ex)
    return new_tag + predict


@app.route("/", methods=['GET', 'POST'])
def root():
    label_list = model.get_labels()
    label_count = len(model.get_labels())
    label_html = ""
    for i in range(label_count):
        label_html+="<li>"+label_list[i]+"</li>"

    out_html = ""
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(FLAGS.upload_folder, filename)
            file.save(file_path)
            print('file saved to %s' % file_path)
            out_html = inference(file_path)

    result = """
    <!doctype html>
    <title>临时测试用</title>
    <h1>自定义训练集图片识别</h1>
    <hr>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='选择图片'>
         <input type=submit value='上传'>
    </form>
    <p>%s</p>
    <h2>目前支持预测的分类有%d种：</h2>
    <ul>%s
    </ul>
    """ % (out_html,label_count,label_html)
    return result


if __name__ == '__main__':
    # 方式一:
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload_folder', type=str, default='./upload/',
                        help='upload_folder path')

    parser.add_argument('--port', type=str, default=5001,
                        help='server with port,if no port, use deault port 5001')

    parser.add_argument('--debug', type=str, default=False,
                        help='is debug?')

    FLAGS, _ = parser.parse_known_args()

    # 方式二
    # tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
    # tf.app.run(main=main)
    app._static_folder = FLAGS.upload_folder
    print('listening on port %d' % FLAGS.port)
    app.run(host='0.0.0.0', port=FLAGS.port, debug=FLAGS.debug, threaded=True)
