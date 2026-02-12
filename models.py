from app import db
from datetime import datetime

class DataFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)  # video or image
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='uploaded')  # uploaded, annotated, processed
    annotations = db.relationship('Annotation', backref='data_file', lazy=True)

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data_file_id = db.Column(db.Integer, db.ForeignKey('data_file.id'), nullable=False)
    timestamp = db.Column(db.Float, nullable=True)  # 视频时间戳
    behavior = db.Column(db.String(100), nullable=False)  # 教学行为类型
    coordinates = db.Column(db.String(255), nullable=True)  # 目标坐标
    annotator = db.Column(db.String(100), default='system')
    annotation_time = db.Column(db.DateTime, default=datetime.utcnow)

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(255), nullable=False)
    model_path = db.Column(db.String(255), nullable=False)
    training_time = db.Column(db.DateTime, default=datetime.utcnow)
    training_data_size = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)

class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=False)
    data_file_id = db.Column(db.Integer, db.ForeignKey('data_file.id'), nullable=False)
    evaluation_time = db.Column(db.DateTime, default=datetime.utcnow)
    correct_predictions = db.Column(db.Integer, nullable=False)
    total_predictions = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)

class TeachingBehavior(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)  # 英文键
    value = db.Column(db.String(100), nullable=False)  # 中文值
    description = db.Column(db.Text, nullable=True)  # 行为描述
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
