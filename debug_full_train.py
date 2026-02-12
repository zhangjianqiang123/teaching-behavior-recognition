from app import app, db
from models import Model, DataFile, Annotation
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

with app.app_context():
    # 获取已标注的数据
    annotated_files = DataFile.query.filter_by(status='annotated').all()
    
    # 准备训练数据
    X = []
    y = []
    
    # 处理每个已标注的文件
    for file in annotated_files:
        annotations = Annotation.query.filter_by(data_file_id=file.id).all()
        
        # 构建视频文件路径
        video_path = None
        if os.path.isabs(file.filepath):
            video_path = file.filepath
        else:
            video_path = os.path.join(os.getcwd(), file.filepath)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            video_path = os.path.join(os.getcwd(), 'static', 'uploads', file.filename)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        
        # 处理每个标注
        for annotation in annotations:
            # 设置视频位置到标注的帧
            frame_index = int(annotation.timestamp)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # 调整大小并提取特征
                img_resized = cv2.resize(frame, (64, 64))
                features = img_resized.flatten()
                X.append(features)
                y.append(annotation.behavior)
        
        # 释放视频捕获对象
        cap.release()
    
    print(f'特征数量: {len(X)}')
    print(f'标签数量: {len(y)}')
    
    # 转换标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f'训练集大小: {len(X_train)}')
    print(f'测试集大小: {len(X_test)}')
    
    # 训练模型
    model = SVC(kernel='linear', probability=True)
    print('开始训练模型...')
    model.fit(X_train, y_train)
    print('模型训练完成')
    
    # 评估模型
    accuracy = model.score(X_test, y_test)
    print(f'模型准确率: {accuracy:.2f}')
    
    # 保存模型
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    model_name = f'teaching_behavior_model_{timestamp}'
    model_path = os.path.join('models', f'{model_name}.joblib')
    
    # 确保models目录存在
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存模型
    joblib.dump({'model': model, 'label_encoder': label_encoder}, model_path)
    print(f'模型保存到: {model_path}')
    
    # 保存模型信息到数据库
    new_model = Model(
        model_name=model_name,
        model_path=model_path,
        training_data_size=len(X),
        accuracy=accuracy
    )
    db.session.add(new_model)
    db.session.commit()
    print(f'模型信息保存到数据库，模型ID: {new_model.id}')
    
    print('完整训练过程成功完成！')