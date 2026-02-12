from app import app, db
from models import Model, DataFile, Annotation
import os

with app.app_context():
    # 检查已标注的数据数量
    annotated_count = DataFile.query.filter_by(status='annotated').count()
    print(f'已标注数据文件数量: {annotated_count}')
    
    # 检查所有已标注的数据文件
    annotated_files = DataFile.query.filter_by(status='annotated').all()
    print(f'已标注数据文件: {[f.filename for f in annotated_files]}')
    
    # 检查标注数量
    total_annotations = sum(len(file.annotations) for file in annotated_files)
    print(f'总标注数量: {total_annotations}')
    
    # 检查标注详情
    for file in annotated_files:
        print(f'文件 {file.filename} 的标注:')
        for annotation in file.annotations:
            print(f'  - 行为: {annotation.behavior}, 时间戳: {annotation.timestamp}, 坐标: {annotation.coordinates}')
    
    # 检查models目录
    if not os.path.exists('models'):
        os.makedirs('models')
    print(f'models目录存在: {os.path.exists('models')}')
    print(f'models目录内容: {os.listdir('models')}')
    
    # 检查数据库中的模型数量
    model_count = Model.query.count()
    print(f'数据库中的模型数量: {model_count}')
    print(f'数据库中的模型: {[m.model_name for m in Model.query.all()]}')