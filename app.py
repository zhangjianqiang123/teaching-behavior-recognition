from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///teaching_behavior.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 创建上传目录
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 创建标注目录
if not os.path.exists('annotations'):
    os.makedirs('annotations')

# 创建模型目录
if not os.path.exists('models'):
    os.makedirs('models')

# 创建视频帧目录
frames_dir = os.path.join('static', 'frames')
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)
    print(f"创建视频帧目录: {frames_dir}")

db = SQLAlchemy(app)

# 导出变量，供routes.py使用
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']

# 导入路由
from routes import *

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)