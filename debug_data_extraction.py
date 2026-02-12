from app import app, db
from models import DataFile, Annotation
import os
import cv2
import numpy as np

with app.app_context():
    # 获取已标注的数据
    annotated_files = DataFile.query.filter_by(status='annotated').all()
    
    # 准备训练数据
    X = []
    y = []
    
    total_annotations = sum(len(Annotation.query.filter_by(data_file_id=file.id).all()) for file in annotated_files)
    processed_annotations = 0
    
    print(f'总标注数量: {total_annotations}')
    
    for file in annotated_files:
        annotations = Annotation.query.filter_by(data_file_id=file.id).all()
        
        print(f'处理文件: {file.filename}')
        print(f'文件类型: {file.file_type}')
        print(f'文件路径: {file.filepath}')
        
        # 构建视频文件路径
        video_path = None
        if os.path.isabs(file.filepath):
            video_path = file.filepath
        else:
            # 基于应用根目录构建绝对路径
            video_path = os.path.join(os.getcwd(), file.filepath)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            # 尝试使用static/uploads目录构建路径
            video_path = os.path.join('static', 'uploads', file.filename)
            if not os.path.exists(video_path):
                # 尝试使用当前工作目录的static/uploads构建路径
                video_path = os.path.join(os.getcwd(), 'static', 'uploads', file.filename)
        
        print(f'实际使用的视频路径: {video_path}')
        print(f'视频文件存在: {os.path.exists(video_path)}')
        
        # 对于视频文件，只打开一次，处理完所有标注后再关闭
        cap = None
        if file.file_type == 'video':
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f'无法打开视频文件: {video_path}')
                continue
            # 获取视频帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # 默认帧率
            print(f'视频帧率: {fps}')
        
        for annotation in annotations:
            # 提取特征
            if file.file_type == 'video':
                # 对于视频，使用已经打开的cap对象提取帧特征
                if cap is not None:
                    # 设置视频位置到标注的帧
                    if annotation.timestamp is not None:
                        print(f'  处理标注: 行为={annotation.behavior}, 时间戳={annotation.timestamp}')
                        # annotation.timestamp 存储的是帧索引，直接使用
                        frame_index = int(annotation.timestamp)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = cap.read()
                        if ret:
                            print(f'    成功读取帧: {frame_index}')
                            # 调整大小并提取特征
                            img_resized = cv2.resize(frame, (64, 64))
                            features = img_resized.flatten()
                            X.append(features)
                            y.append(annotation.behavior)
                            print(f'    成功提取特征，特征长度: {len(features)}')
                        else:
                            print(f'    读取帧失败: {frame_index}')
                    else:
                        print(f'    标注没有帧索引，跳过')
            
        # 处理完所有标注后，释放视频捕获对象
        if cap is not None:
            cap.release()
        
        # 更新处理进度
        processed_annotations += len(annotations)
        print(f'处理进度: {processed_annotations}/{total_annotations}')
    
    print(f'\n数据提取结果:')
    print(f'X的长度: {len(X)}')
    print(f'y的长度: {len(y)}')
    
    if X:
        print(f'X的形状: {np.array(X).shape}')
        print(f'y的内容: {y}')
    else:
        print(f'没有提取到任何特征！')