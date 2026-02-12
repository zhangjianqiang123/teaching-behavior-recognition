from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from app import app, db, UPLOAD_FOLDER
from models import DataFile, Annotation, Model, Evaluation
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import base64
from datetime import datetime

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# 检查文件扩展名

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 从数据库中获取教学行为类型
from models import TeachingBehavior
import threading
import time

def get_behaviors():
    """从数据库中获取教学行为类型，按id倒序排列"""
    # 按id倒序获取教学行为
    behaviors = TeachingBehavior.query.order_by(TeachingBehavior.id.desc()).all()
    # 返回一个列表，包含(key, value)元组，保持按id倒序
    return [(behavior.key, behavior.value) for behavior in behaviors]

# 获取当前的教学行为类型，用于模板渲染
@app.context_processor
def inject_behaviors():
    """将有序的教学行为列表注入到所有模板中"""
    return {'behaviors_list': get_behaviors()}

# 初始化默认教学行为
def init_behaviors():
    """初始化默认教学行为"""
    default_behaviors = {
        'lecturing': '讲授',
        'questioning': '提问',
        'group_discussion': '小组讨论',
        'individual_work': '个人作业',
        'demonstration': '演示',
        'interaction': '互动',
        'assessment': '评估',
        'other': '其他'
    }
    
    for key, value in default_behaviors.items():
        if not TeachingBehavior.query.filter_by(key=key).first():
            behavior = TeachingBehavior(key=key, value=value)
            db.session.add(behavior)
    db.session.commit()

# 全局BEHAVIORS字典，用于非模板渲染的地方
BEHAVIORS = {}

# 训练进度跟踪
TRAINING_STATUS = {
    'progress': 0,
    'status': '未开始训练',
    'running': False
}

# 训练线程锁
TRAINING_LOCK = threading.Lock()

# 延迟初始化，将在需要时动态获取
# 获取当前的教学行为类型
def update_behaviors():
    """更新全局BEHAVIORS字典"""
    global BEHAVIORS
    # 从get_behaviors()返回的列表重新构建字典
    behaviors_list = get_behaviors()
    BEHAVIORS = {key: value for key, value in behaviors_list}

# 在应用上下文中初始化
with app.app_context():
    init_behaviors()
    update_behaviors()

# 清空数据
@app.route('/clear_data', methods=['POST'])
def clear_data():
    import shutil
    
    # 删除所有评估结果
    Evaluation.query.delete()
    # 删除所有模型
    # 删除模型文件
    for model in Model.query.all():
        if os.path.exists(model.model_path):
            try:
                os.remove(model.model_path)
            except:
                pass
    Model.query.delete()
    # 删除所有标注
    Annotation.query.delete()
    # 删除所有数据文件
    # 删除上传的文件
    for data_file in DataFile.query.all():
        if os.path.exists(data_file.filepath):
            try:
                os.remove(data_file.filepath)
            except:
                pass
    DataFile.query.delete()
    
    # 删除所有视频帧目录
    frames_root_dir = os.path.join('static', 'frames')
    if os.path.exists(frames_root_dir):
        try:
            # 删除frames目录下的所有子目录
            for item in os.listdir(frames_root_dir):
                item_path = os.path.join(frames_root_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        except:
            pass
    
    # 提交事务
    db.session.commit()
    flash('数据已成功清空！')
    return redirect(url_for('index'))

# 主页
@app.route('/')
def index():
    # 统计数据
    total_files = DataFile.query.count()
    annotated_files = DataFile.query.filter_by(status='annotated').count()
    total_models = Model.query.count()
    latest_model = Model.query.order_by(Model.training_time.desc()).first()
    latest_accuracy = "{:.2f}".format(latest_model.accuracy * 100) if latest_model else "0.00"
    
    return render_template('index.html', 
                           total_files=total_files, 
                           annotated_files=annotated_files, 
                           total_models=total_models, 
                           latest_accuracy=latest_accuracy)

# 数据上传页面
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # 保存文件
            filename = file.filename
            # 使用app.config['UPLOAD_FOLDER']确保路径正确
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                flash('文件上传成功！')
            except Exception as e:
                flash(f'文件保存失败: {str(e)}')
                return redirect(request.url)
            
            # 确定文件类型
            file_ext = filename.rsplit('.', 1)[1].lower()
            if file_ext in {'mp4', 'avi', 'mov'}:
                file_type = 'video'
            else:
                file_type = 'image'
            
            # 保存到数据库
            data_file = DataFile(
                filename=filename,
                filepath=filepath,
                file_type=file_type
            )
            db.session.add(data_file)
            db.session.commit()
            
            flash('File uploaded successfully!')
            return redirect(url_for('annotate', file_id=data_file.id))
    return render_template('upload.html')

# 提取视频帧并保存
def extract_video_frames(video_path, output_dir, interval=1):
    """
    提取视频帧并保存到指定目录
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param interval: 帧间隔，默认为1秒
    :return: 提取的帧数量
    """
    # 写入日志
    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n=== 视频帧提取函数开始 ===\n")
        log_file.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"视频路径: {video_path}\n")
        log_file.write(f"输出目录: {output_dir}\n")
        log_file.write(f"帧间隔: {interval} 秒\n")
        log_file.write(f"当前工作目录: {os.getcwd()}\n")
    
    print(f"=== 视频帧提取函数开始 ===")
    print(f"视频路径: {video_path}")
    print(f"输出目录: {output_dir}")
    print(f"帧间隔: {interval} 秒")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查视频文件是否存在
    # 写入日志
    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
        log_file.write(f"检查视频文件是否存在: {os.path.exists(video_path)}\n")
    
    if not os.path.exists(video_path):
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"错误: 视频文件不存在: {video_path}\n")
        print(f"错误: 视频文件不存在: {video_path}")
        raise Exception(f'视频文件不存在: {video_path}')
    
    # 检查视频文件大小
    file_size = os.path.getsize(video_path)
    # 写入日志
    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
        log_file.write(f"视频文件大小: {file_size} bytes\n")
    
    print(f"视频文件大小: {file_size} bytes")
    if file_size == 0:
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"错误: 视频文件为空: {video_path}\n")
        print(f"错误: 视频文件为空: {video_path}")
        raise Exception(f'视频文件为空: {video_path}')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"创建输出目录成功: {output_dir}")
        except Exception as e:
            print(f"错误: 创建帧目录失败: {str(e)}")
            raise Exception(f'创建帧目录失败: {str(e)}')
    
    # 检查目录权限
    print(f"目录可写性: {os.access(output_dir, os.W_OK)}")
    print(f"目录存在: {os.path.exists(output_dir)}")
    
    # 尝试打开视频文件
    print(f"尝试打开视频文件...")
    cap = None
    try:
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"尝试打开视频文件: {video_path}\n")
        
        cap = cv2.VideoCapture(video_path)
        
        # 检查视频是否打开成功
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {video_path}")
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"错误: 无法打开视频文件: {video_path}\n")
            # 尝试使用绝对路径
            abs_path = os.path.abspath(video_path)
            print(f"尝试使用绝对路径: {abs_path}")
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"尝试使用绝对路径: {abs_path}\n")
            cap = cv2.VideoCapture(abs_path)
            if not cap.isOpened():
                print(f"错误: 仍然无法打开视频文件: {abs_path}")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"错误: 仍然无法打开视频文件: {abs_path}\n")
                # 尝试使用不同的打开方式
                print("尝试使用其他方式打开视频文件...")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"尝试使用其他方式打开视频文件...\n")
                # 尝试使用cv2.CAP_FFMPEG
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print(f"错误: 无法使用FFMPEG打开视频文件: {video_path}")
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"错误: 无法使用FFMPEG打开视频文件: {video_path}\n")
                    raise Exception(f'无法打开视频文件: {video_path}')
        
        print(f"视频文件打开成功")
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"视频文件打开成功\n")
        
        # 获取视频属性
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"视频总帧数: {total_frames}")
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"视频总帧数: {total_frames}\n")
        except Exception as e:
            print(f"获取总帧数失败: {str(e)}")
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"获取总帧数失败: {str(e)}\n")
            total_frames = -1
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # 默认帧率
                print(f"视频帧率未知，使用默认帧率: {fps}")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"视频帧率未知，使用默认帧率: {fps}\n")
            else:
                print(f"视频帧率: {fps}")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"视频帧率: {fps}\n")
        except Exception as e:
            print(f"获取帧率失败: {str(e)}")
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"获取帧率失败: {str(e)}\n")
            fps = 30  # 默认帧率
        
        # 计算帧间隔
        frame_interval = int(fps * interval)
        if frame_interval <= 0:
            frame_interval = 1  # 确保至少提取一帧
        print(f"计算的帧间隔: {frame_interval} 帧")
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"计算的帧间隔: {frame_interval} 帧\n")
        
        frame_count = 0
        extracted_count = 0
        
        print("开始读取视频帧...")
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"开始读取视频帧...\n")
        
        # 处理视频帧
        while cap.isOpened():
            ret, frame = cap.read()
            
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"读取帧 {frame_count}: ret={ret}, frame={frame is not None}\n")
            
            if not ret:
                print(f"读取视频帧结束，ret={ret}")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"读取视频帧结束，ret={ret}\n")
                break
            
            # 检查帧是否为空
            if frame is None:
                print(f"警告: 第 {frame_count} 帧为空，跳过")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"警告: 第 {frame_count} 帧为空，跳过\n")
                frame_count += 1
                continue
            
            # 根据帧间隔保存帧
            if frame_count % frame_interval == 0:
                # 使用绝对路径保存帧文件
                frame_filename = os.path.join(output_dir, f'frame_{extracted_count:04d}.jpg')
                abs_frame_filename = os.path.abspath(frame_filename)
                
                try:
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"尝试保存帧 {frame_count}: {abs_frame_filename}\n")
                    
                    # 保存帧文件
                    print(f"保存帧 {frame_count}: {abs_frame_filename}")
                    success = cv2.imwrite(abs_frame_filename, frame)
                    
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"保存帧结果: {success}\n")
                    
                    if success:
                        print(f"保存帧成功: {abs_frame_filename}")
                        # 写入日志
                        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"保存帧成功: {abs_frame_filename}\n")
                        extracted_count += 1
                    else:
                        print(f"保存帧失败: {abs_frame_filename}")
                        # 写入日志
                        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"保存帧失败: {abs_frame_filename}\n")
                        # 检查目录权限
                        print(f"目录可写性: {os.access(output_dir, os.W_OK)}")
                        print(f"目录存在: {os.path.exists(output_dir)}")
                        print(f"目录权限: {oct(os.stat(output_dir).st_mode)[-3:]}")
                        # 写入日志
                        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"目录可写性: {os.access(output_dir, os.W_OK)}\n")
                            log_file.write(f"目录存在: {os.path.exists(output_dir)}\n")
                            log_file.write(f"目录权限: {oct(os.stat(output_dir).st_mode)[-3:]}\n")
                except Exception as e:
                    print(f"保存帧时发生错误: {str(e)}")
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"保存帧时发生错误: {str(e)}\n")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"详细错误: {traceback.format_exc()}\n")
            
            frame_count += 1
            
            # 打印进度
            print(f"已处理 {frame_count} 帧，已提取 {extracted_count} 帧")
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"已处理 {frame_count} 帧，已提取 {extracted_count} 帧\n")
            
            # 根据视频总帧数和帧率计算视频长度（秒）
            video_length = total_frames / fps if total_frames > 0 else 0
            # 根据视频长度和帧间隔计算理论最大提取帧数
            # 每3秒提取1帧，同时设置一个合理的上限，避免处理过多
            max_frames = int(video_length / interval) + 1
            max_frames = min(max_frames, 200)  # 最多提取200帧，可根据需求调整
            
            # 限制提取的帧数量
            if extracted_count >= max_frames:
                print(f"达到最大提取帧数 {max_frames}，停止处理")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"达到最大提取帧数 {max_frames}，停止处理\n")
                break
            
    except Exception as e:
        print(f"处理视频时发生错误: {str(e)}")
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"处理视频时发生错误: {str(e)}\n")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"详细错误: {traceback.format_exc()}\n")
        raise
    finally:
        if cap is not None:
            cap.release()
            print("释放视频捕获对象")
            # 写入日志
            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"释放视频捕获对象\n")
    
    # 检查输出目录中的文件
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"=== 提取完成 ===")
        print(f"输出目录中的文件数量: {len(files)}")
        print(f"输出目录中的文件: {files}")
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"输出目录中的文件数量: {len(files)}\n")
            log_file.write(f"输出目录中的文件: {files}\n")
    else:
        print(f"错误: 输出目录不存在: {output_dir}")
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"错误: 输出目录不存在: {output_dir}\n")
    
    print(f"视频帧提取完成，共提取 {extracted_count} 帧")
    # 写入日志
    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
        log_file.write(f"视频帧提取完成，共提取 {extracted_count} 帧\n")
    return extracted_count

# 数据标注页面
@app.route('/annotate/<int:file_id>', methods=['GET', 'POST'])
def annotate(file_id):
    data_file = DataFile.query.get_or_404(file_id)
    page = request.args.get('page', 1, type=int)
    per_page = 5
    
    if request.method == 'POST':
        behavior = request.form.get('behavior')
        timestamp = request.form.get('timestamp')
        coordinates = request.form.get('coordinates')
        frame_index = request.form.get('frame_index')
        
        # 创建标注
        annotation = Annotation(
            data_file_id=file_id,
            timestamp=float(timestamp) if timestamp else None,
            behavior=behavior,
            coordinates=coordinates
        )
        db.session.add(annotation)
        db.session.commit()
        
        # 更新文件状态
        data_file.status = 'annotated'
        db.session.commit()
        
        flash('Annotation saved successfully!')
        return redirect(url_for('annotate', file_id=file_id, page=page))
    
    # 分页获取标注
    annotations = Annotation.query.filter_by(data_file_id=file_id).paginate(page=page, per_page=per_page, error_out=False)
    
    # 为每个标注添加缩略图路径
    for annotation in annotations.items:
        if data_file.file_type == 'video' and annotation.timestamp is not None:
            # 为视频标注添加缩略图路径
            frame_file = f'frame_{int(annotation.timestamp):04d}.jpg'
            annotation.thumbnail_path = f'frames/{file_id}/{frame_file}'
        elif data_file.file_type == 'image':
            # 为图片标注添加缩略图路径
            annotation.thumbnail_path = f'uploads/{data_file.filename}'

    # 处理视频文件，提取帧
    frames = []
    current_frame = None
    total_frames = 0

    if data_file.file_type == 'video':
        # 视频帧保存目录（使用文件ID作为目录名，避免中文路径问题）
        frames_dir = os.path.join('static', 'frames', str(data_file.id))
        
        # 写入日志
        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"\n=== 开始处理视频 ===\n")
            log_file.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"视频文件名: {data_file.filename}\n")
            log_file.write(f"视频文件类型: {data_file.file_type}\n")
            log_file.write(f"视频文件路径: {data_file.filepath}\n")
            log_file.write(f"帧保存目录: {frames_dir}\n")
        
        # 打印调试信息
        print(f"=== 开始处理视频 ===")
        print(f"视频文件名: {data_file.filename}")
        print(f"视频文件类型: {data_file.file_type}")
        print(f"视频文件路径: {data_file.filepath}")
        print(f"帧保存目录: {frames_dir}")
        
        # 确保frames主目录存在
        main_frames_dir = os.path.join('static', 'frames')
        if not os.path.exists(main_frames_dir):
            try:
                os.makedirs(main_frames_dir)
                print(f"创建主帧目录成功: {main_frames_dir}")
            except Exception as e:
                print(f"创建主帧目录失败: {str(e)}")
                flash(f'创建主帧目录失败: {str(e)}')
                return render_template('annotate.html', 
                                       data_file=data_file, 
                                       behaviors=BEHAVIORS, 
                                       annotations=annotations, 
                                       frames=frames, 
                                       page=page, 
                                       total_pages=0, 
                                       total_frames=0)
        
        # 确保帧目录存在
        if not os.path.exists(frames_dir):
            try:
                os.makedirs(frames_dir)
                print(f"创建帧目录成功: {frames_dir}")
            except Exception as e:
                print(f"创建帧目录失败: {str(e)}")
                flash(f'创建帧目录失败: {str(e)}')
                return render_template('annotate.html', 
                                       data_file=data_file, 
                                       behaviors=BEHAVIORS, 
                                       annotations=annotations, 
                                       frames=frames, 
                                       page=page, 
                                       total_pages=0, 
                                       total_frames=0)
        
        # 检查帧目录是否创建成功
        if not os.path.exists(frames_dir):
            print(f"错误: 帧目录创建失败，仍然不存在: {frames_dir}")
            flash(f'帧目录创建失败')
            return render_template('annotate.html', 
                                   data_file=data_file, 
                                   behaviors=BEHAVIORS, 
                                   annotations=annotations, 
                                   frames=frames, 
                                   page=page, 
                                   total_pages=0, 
                                   total_frames=0)
        
        # 打印当前工作目录
        print(f"当前工作目录: {os.getcwd()}")
        print(f"frames目录绝对路径: {os.path.abspath(frames_dir)}")
        
        # 检查帧目录是否存在，以及是否有帧文件
        try:
            # 构建正确的视频文件路径
            print(f"=== 构建视频文件路径 ===")
            print(f"数据库中的filepath: {data_file.filepath}")
            print(f"是否是绝对路径: {os.path.isabs(data_file.filepath)}")
            
            # 尝试多种路径构建方式
            possible_paths = []
            
            # 1. 直接使用数据库中的路径
            if os.path.isabs(data_file.filepath):
                possible_paths.append(data_file.filepath)
            else:
                # 2. 基于应用根目录构建绝对路径
                possible_paths.append(os.path.join(os.getcwd(), data_file.filepath))
            
            # 3. 基于static/uploads目录构建路径
            possible_paths.append(os.path.join('static', 'uploads', data_file.filename))
            
            # 4. 基于当前工作目录的static/uploads构建路径
            possible_paths.append(os.path.join(os.getcwd(), 'static', 'uploads', data_file.filename))
            
            # 5. 基于应用根目录的static/uploads构建绝对路径
            possible_paths.append(os.path.abspath(os.path.join('static', 'uploads', data_file.filename)))
            
            # 打印所有可能的路径
            for i, path in enumerate(possible_paths):
                exists = os.path.exists(path)
                print(f"路径 {i+1}: {path} (存在: {exists})")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"路径 {i+1}: {path} (存在: {exists})\n")
                
                if exists:
                    file_size = os.path.getsize(path)
                    print(f"  文件大小: {file_size} bytes")
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"  文件大小: {file_size} bytes\n")
                    video_path = path
                    break
            else:
                # 所有路径都不存在
                print(f"错误: 找不到视频文件")
                flash(f'视频文件不存在: {data_file.filename}')
                return render_template('annotate.html', 
                                       data_file=data_file, 
                                       behaviors=BEHAVIORS, 
                                       annotations=annotations, 
                                       frames=frames, 
                                       page=page, 
                                       total_pages=0, 
                                       total_frames=0)
            
            print(f"选择的视频路径: {video_path}")
            
            # 打印调试信息
            print(f"=== 调试信息 ===")
            print(f"视频文件名: {data_file.filename}")
            print(f"视频文件路径: {data_file.filepath}")
            print(f"构建的视频路径: {video_path}")
            print(f"视频文件存在: {os.path.exists(video_path)}")
            print(f"帧目录: {frames_dir}")
            print(f"帧目录存在: {os.path.exists(frames_dir)}")
            
            # 确保帧目录存在
            if not os.path.exists(frames_dir):
                try:
                    os.makedirs(frames_dir)
                    print(f"创建帧目录成功: {frames_dir}")
                except Exception as e:
                    print(f"创建帧目录失败: {str(e)}")
                    flash(f'创建帧目录失败: {str(e)}')
                    return render_template('annotate.html', 
                                           data_file=data_file, 
                                           behaviors=BEHAVIORS, 
                                           annotations=annotations, 
                                           frames=frames, 
                                           page=page, 
                                           total_pages=0, 
                                           total_frames=0)
            
            # 获取已提取的帧文件列表
            frame_files = []
            if os.path.exists(frames_dir):
                print(f"读取帧目录: {frames_dir}")
                try:
                    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
                    # 按文件名排序
                    frame_files.sort()
                    # 计算已提取的帧数量
                    total_frames = len(frame_files)
                    print(f"已提取的帧数量: {total_frames}")
                    print(f"帧文件列表: {frame_files}")
                except Exception as e:
                    print(f"读取帧目录失败: {str(e)}")
                    total_frames = 0
            
            # 如果帧文件为空，提取视频帧
            if total_frames == 0:
                print(f"=== 开始提取视频帧 ===")
                # 写入日志
                with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n=== 开始提取视频帧 ===\n")
                    log_file.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"视频路径: {video_path}\n")
                    log_file.write(f"输出目录: {frames_dir}\n")
                    log_file.write(f"视频文件存在: {os.path.exists(video_path)}\n")
                    log_file.write(f"视频文件大小: {os.path.getsize(video_path)} bytes\n")
                
                try:
                    print(f"视频路径: {video_path}")
                    print(f"输出目录: {frames_dir}")
                    print(f"视频文件存在: {os.path.exists(video_path)}")
                    print(f"视频文件大小: {os.path.getsize(video_path)} bytes")
                    
                    # 直接调用视频帧提取函数
                    print(f"调用extract_video_frames函数")
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"调用extract_video_frames函数\n")
                    
                    # 每3秒提取1帧
                    extracted_count = extract_video_frames(video_path, frames_dir, interval=3)
                    
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"视频帧提取函数返回: {extracted_count}\n")
                    
                    print(f"视频帧提取函数返回: {extracted_count}")
                    
                    # 提取视频帧后，获取帧文件列表
                    if os.path.exists(frames_dir):
                        print(f"读取提取后的帧目录: {frames_dir}")
                        # 写入日志
                        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"读取提取后的帧目录: {frames_dir}\n")
                        
                        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
                        # 按文件名排序
                        frame_files.sort()
                        total_frames = len(frame_files)
                        
                        # 写入日志
                        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"提取后帧数量: {total_frames}\n")
                            log_file.write(f"提取后帧文件列表: {frame_files}\n")
                        
                        print(f"提取后帧数量: {total_frames}")
                        print(f"提取后帧文件列表: {frame_files}")
                        
                        # 如果仍然没有帧文件，尝试使用不同的帧间隔
                        if total_frames == 0:
                            print("=== 尝试使用不同的帧间隔 ===")
                            print("尝试帧间隔为0.5秒")
                            # 写入日志
                            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                                log_file.write("=== 尝试使用不同的帧间隔 ===\n")
                                log_file.write("尝试帧间隔为0.5秒\n")
                            
                            extracted_count = extract_video_frames(video_path, frames_dir, interval=0.5)
                            
                            # 写入日志
                            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                                log_file.write(f"新的提取结果: {extracted_count}\n")
                            
                            print(f"新的提取结果: {extracted_count}")
                            
                            # 再次检查帧文件
                            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
                            frame_files.sort()
                            total_frames = len(frame_files)
                            
                            # 写入日志
                            with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                                log_file.write(f"新的帧数量: {total_frames}\n")
                                log_file.write(f"新的帧文件列表: {frame_files}\n")
                            
                            print(f"新的帧数量: {total_frames}")
                            print(f"新的帧文件列表: {frame_files}")
                    else:
                        print(f"错误: 提取后帧目录不存在")
                        # 写入日志
                        with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"错误: 提取后帧目录不存在\n")
                except Exception as e:
                    print(f"提取视频帧失败: {str(e)}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 写入日志
                    with open('video_extract.log', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"提取视频帧失败: {str(e)}\n")
                        log_file.write(f"详细错误: {traceback.format_exc()}\n")
                    
                    flash(f'提取视频帧失败: {str(e)}')
                    return render_template('annotate.html', 
                                           data_file=data_file, 
                                           behaviors=BEHAVIORS, 
                                           annotations=annotations, 
                                           frames=frames, 
                                           page=page, 
                                           total_pages=0, 
                                           total_frames=0)
        except Exception as e:
            flash(f'检查帧目录失败: {str(e)}')
            return render_template('annotate.html', 
                                   data_file=data_file, 
                                   behaviors=BEHAVIORS, 
                                   annotations=annotations, 
                                   frames=frames, 
                                   page=page, 
                                   total_pages=0, 
                                   total_frames=0)
        
        # 分页获取帧
        frame_per_page = 8
        start_frame = (page - 1) * frame_per_page
        end_frame = start_frame + frame_per_page
        
        # 遍历实际存在的帧文件
        for i, frame_file in enumerate(frame_files[start_frame:end_frame]):
            # 从文件名中提取帧索引
            frame_index = int(frame_file.split('_')[1].split('.')[0])
            # 使用正斜杠构建URL路径，确保跨平台兼容性
            frame_path = f'frames/{data_file.id}/{frame_file}'
            # 检查该帧是否已标注
            is_annotated = any(ann.timestamp == frame_index for ann in annotations.items)
            frames.append({
                'index': frame_index,
                'path': frame_path,
                'is_annotated': is_annotated
            })
        
        # 计算总页数
        total_pages = (total_frames + frame_per_page - 1) // frame_per_page
    else:
        total_frames = 0
        total_pages = 0
    
    return render_template('annotate.html', 
                           data_file=data_file, 
                           behaviors=BEHAVIORS, 
                           annotations=annotations, 
                           frames=frames, 
                           page=page, 
                           total_pages=total_pages, 
                           total_frames=total_frames)

# 训练模型的实际执行函数
def train_model():
    """实际执行模型训练的函数"""
    global TRAINING_STATUS
    
    # 在应用上下文中执行训练
    with app.app_context():
        try:
            # 更新训练状态
            with TRAINING_LOCK:
                TRAINING_STATUS['progress'] = 10
                TRAINING_STATUS['status'] = '正在获取已标注数据...'
        
            # 获取所有已标注的数据
            annotated_files = DataFile.query.filter_by(status='annotated').all()
        
            if not annotated_files:
                with TRAINING_LOCK:
                    TRAINING_STATUS['progress'] = 0
                    TRAINING_STATUS['status'] = '没有可用的已标注数据'
                    TRAINING_STATUS['running'] = False
                return
        
            # 更新训练状态
            with TRAINING_LOCK:
                TRAINING_STATUS['progress'] = 20
                TRAINING_STATUS['status'] = '正在准备训练数据...'
        
            # 准备训练数据
            X = []
            y = []
        
            total_annotations = sum(len(Annotation.query.filter_by(data_file_id=file.id).all()) for file in annotated_files)
            processed_annotations = 0
        
            for file in annotated_files:
                annotations = Annotation.query.filter_by(data_file_id=file.id).all()
                
                for annotation in annotations:
                    # 提取特征
                    if file.file_type == 'image':
                        # 直接从图片文件提取特征
                        img = cv2.imread(file.filepath)
                        if img is not None:
                            # 调整大小并提取特征
                            img_resized = cv2.resize(img, (64, 64))
                            features = img_resized.flatten()
                            X.append(features)
                            y.append(annotation.behavior)
                    elif file.file_type == 'video':
                        # 使用已经提取的帧图片，而不是重新从视频中提取
                        if annotation.timestamp is not None:
                            # 帧图片保存目录（使用文件ID作为目录名）
                            frames_dir = os.path.join('static', 'frames', str(file.id))
                            frame_index = int(annotation.timestamp)
                            # 构建帧图片路径
                            frame_file = os.path.join(frames_dir, f'frame_{frame_index:04d}.jpg')
                            
                            # 检查帧图片是否存在
                            if os.path.exists(frame_file):
                                # 直接从帧图片中提取特征
                                img = cv2.imread(frame_file)
                                if img is not None:
                                    # 调整大小并提取特征
                                    img_resized = cv2.resize(img, (64, 64))
                                    features = img_resized.flatten()
                                    X.append(features)
                                    y.append(annotation.behavior)
                                    print(f"成功从帧图片提取特征: {frame_file}")
                                else:
                                    # 读取图片失败，记录日志
                                    print(f"读取帧图片失败: {frame_file}")
                            else:
                                # 帧图片不存在，记录日志
                                print(f"帧图片不存在: {frame_file}")
                        else:
                            # 标注没有帧索引，跳过
                            print(f"标注没有帧索引: {file.filename}, 标注ID: {annotation.id}")
                
                # 更新处理进度
                processed_annotations += len(annotations)
                progress = 20 + int(60 * processed_annotations / total_annotations)
                with TRAINING_LOCK:
                    TRAINING_STATUS['progress'] = progress
                    TRAINING_STATUS['status'] = f'正在处理数据... ({processed_annotations}/{total_annotations})'
        
            if not X:
                with TRAINING_LOCK:
                    TRAINING_STATUS['progress'] = 0
                    TRAINING_STATUS['status'] = '从已标注数据中提取特征失败'
                    TRAINING_STATUS['running'] = False
                return
        
            # 更新训练状态
            with TRAINING_LOCK:
                TRAINING_STATUS['progress'] = 80
                TRAINING_STATUS['status'] = '正在训练模型...'
        
            # 转换标签
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
            # 训练模型，使用更快的算法
            model = SVC(kernel='linear', probability=True, cache_size=500)  # 增加缓存大小
            model.fit(X_train, y_train)
        
            # 更新训练状态
            with TRAINING_LOCK:
                TRAINING_STATUS['progress'] = 90
                TRAINING_STATUS['status'] = '正在评估模型...'
        
            # 评估模型
            accuracy = model.score(X_test, y_test)
        
            # 更新训练状态
            with TRAINING_LOCK:
                TRAINING_STATUS['progress'] = 95
                TRAINING_STATUS['status'] = '正在保存模型...'
        
            # 保存模型
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            model_name = f'teaching_behavior_model_{timestamp}'
            model_path = os.path.join('models', f'{model_name}.joblib')
            
            # 确保models目录存在
            if not os.path.exists('models'):
                os.makedirs('models')
            
            joblib.dump({'model': model, 'label_encoder': label_encoder}, model_path)
            print(f"Model file saved to: {model_path}")
        
            # 保存模型信息到数据库
            new_model = Model(
                model_name=model_name,
                model_path=model_path,
                training_data_size=len(X),
                accuracy=accuracy
            )
            db.session.add(new_model)
            db.session.commit()
            print(f"Model saved to database: {model_name}, accuracy: {accuracy:.2f}")
        
            # 训练完成
            with TRAINING_LOCK:
                TRAINING_STATUS['progress'] = 100
                TRAINING_STATUS['status'] = f'训练完成! 准确率: {accuracy:.2f}'
                TRAINING_STATUS['accuracy'] = accuracy
                TRAINING_STATUS['running'] = False
        except Exception as e:
            # 训练失败，记录详细错误信息
            import traceback
            error_info = traceback.format_exc()
            print(f"Training error: {error_info}")
            with TRAINING_LOCK:
                TRAINING_STATUS['progress'] = 0
                TRAINING_STATUS['status'] = f'训练失败: {str(e)}'
                TRAINING_STATUS['running'] = False

# 训练模型页面
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # 检查是否已有训练在进行
        with TRAINING_LOCK:
            if TRAINING_STATUS['running']:
                return jsonify({'success': False, 'message': '已有训练任务在进行中，请稍后再试'})
            
            # 初始化训练状态
            TRAINING_STATUS['progress'] = 0
            TRAINING_STATUS['status'] = '正在启动训练...'
            TRAINING_STATUS['running'] = True
        
        # 直接执行训练，不使用线程，避免状态管理问题
        train_model()
        
        # 返回训练结果
        with TRAINING_LOCK:
            if TRAINING_STATUS['progress'] == 100:
                return jsonify({'success': True, 'message': '训练完成', 'accuracy': TRAINING_STATUS.get('accuracy', 0)})
            else:
                return jsonify({'success': False, 'message': TRAINING_STATUS['status']})
    
    # 获取已标注的数据统计
    annotated_count = DataFile.query.filter_by(status='annotated').count()
    return render_template('train.html', annotated_count=annotated_count)

# 训练状态查询API
@app.route('/train/status')
def train_status():
    """查询训练状态的API"""
    global TRAINING_STATUS
    
    with TRAINING_LOCK:
        # 返回当前训练状态的副本
        status = TRAINING_STATUS.copy()
    
    return jsonify(status)

# 模型列表页面
@app.route('/models')
def models():
    models = Model.query.order_by(Model.training_time.desc()).all()
    return render_template('models.html', models=models)

# 数据管理页面
@app.route('/data')
def data():
    """数据管理页面，展示所有上传的数据文件"""
    data_files = DataFile.query.order_by(DataFile.upload_time.desc()).all()
    return render_template('data.html', data_files=data_files)

# 教学行为管理页面
@app.route('/behaviors', methods=['GET', 'POST'])
def behaviors():
    """教学行为管理页面，允许增删改查教学行为类型"""
    from flask import request
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add':
            # 添加新的教学行为
            key = request.form.get('key')
            value = request.form.get('value')
            if key and value:
                # 检查是否已存在相同的key
                if not TeachingBehavior.query.filter_by(key=key).first():
                    behavior = TeachingBehavior(key=key, value=value)
                    db.session.add(behavior)
                    db.session.commit()
                    update_behaviors()
                else:
                    flash('该行为类型已存在!')
        elif action == 'update':
            # 更新教学行为
            behavior_id = request.form.get('behavior_id')
            key = request.form.get('key')
            value = request.form.get('value')
            if behavior_id and key and value:
                behavior = TeachingBehavior.query.get(behavior_id)
                if behavior:
                    behavior.key = key
                    behavior.value = value
                    db.session.commit()
                    update_behaviors()
        elif action == 'delete':
            # 删除教学行为
            behavior_id = request.form.get('behavior_id')
            if behavior_id:
                behavior = TeachingBehavior.query.get(behavior_id)
                if behavior:
                    db.session.delete(behavior)
                    db.session.commit()
                    update_behaviors()
    
    # 获取所有教学行为
    all_behaviors = TeachingBehavior.query.all()
    return render_template('behaviors.html', behaviors=all_behaviors)

# 评估模型页面
@app.route('/evaluate/<int:model_id>', methods=['GET', 'POST'])
def evaluate(model_id):
    model = Model.query.get_or_404(model_id)
    
    if request.method == 'POST':
        data_file_id = request.form.get('data_file_id')
        data_file = DataFile.query.get(data_file_id)
        
        if not data_file:
            flash('无效的数据文件!')
            return redirect(request.url)
        
        # 加载模型
        model_data = joblib.load(model.model_path)
        clf = model_data['model']
        label_encoder = model_data['label_encoder']
        
        # 评估数据
        correct_predictions = 0
        total_predictions = 0
        
        # 行为统计
        behavior_counts = {}
        behavior_accuracies = {}
        for behavior in BEHAVIORS.keys():
            behavior_counts[behavior] = 0
            behavior_accuracies[behavior] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0
            }
        
        # 保存每个帧的预测结果
        frame_predictions = []
        
        if data_file.file_type == 'image':
            # 处理图像
            img = cv2.imread(data_file.filepath)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                features = img_resized.flatten().reshape(1, -1)
                prediction = clf.predict(features)
                predicted_behavior = label_encoder.inverse_transform(prediction)[0]
                
                # 统计行为
                behavior_counts[predicted_behavior] += 1
                
                # 保存图像为base64编码，以便在前端显示
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 检查是否有标注
                annotations = Annotation.query.filter_by(data_file_id=data_file_id).all()
                annotation_coordinates = ''
                true_behavior = None
                has_annotation = False
                
                if annotations:
                    annotation_coordinates = annotations[0].coordinates
                    true_behavior = annotations[0].behavior
                    has_annotation = True
                
                # 保存帧预测信息
                frame_predictions.append({
                    'frame_index': 0,
                    'behavior': predicted_behavior,
                    'image_data': f'data:image/jpeg;base64,{img_base64}',
                    'coordinates': annotation_coordinates,
                    'true_behavior': true_behavior
                })
                
                # 更新准确率统计
                if has_annotation:
                    # 统计该行为的总预测次数
                    behavior_accuracies[predicted_behavior]['total'] += 1
                    # 如果预测正确，统计正确次数
                    if predicted_behavior == true_behavior:
                        behavior_accuracies[predicted_behavior]['correct'] += 1
                        correct_predictions += 1
                    total_predictions = 1
        
        elif data_file.file_type == 'video':
            # 处理视频，使用已经提取的帧图片进行评估
            # 帧图片保存目录（使用文件ID作为目录名）
            frames_dir = os.path.join('static', 'frames', str(data_file.id))
            
            # 检查帧目录是否存在
            if os.path.exists(frames_dir):
                # 获取所有帧图片
                frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
                
                # 最多评估100帧
                max_frames = min(100, len(frame_files))
                extracted_frames = 0
                
                # 处理每个帧图片
                for frame_file in frame_files:
                    if extracted_frames >= max_frames:
                        break
                    
                    # 解析帧索引
                    try:
                        frame_index = int(frame_file.split('_')[1].split('.')[0])
                    except ValueError:
                        continue
                    
                    # 构建帧图片路径
                    frame_path = os.path.join(frames_dir, frame_file)
                    
                    # 读取帧图片
                    img = cv2.imread(frame_path)
                    if img is not None:
                        # 提取特征
                        img_resized = cv2.resize(img, (64, 64))
                        features = img_resized.flatten().reshape(1, -1)
                        
                        # 进行预测
                        prediction = clf.predict(features)
                        predicted_behavior = label_encoder.inverse_transform(prediction)[0]
                        
                        # 统计行为
                        behavior_counts[predicted_behavior] += 1
                        
                        # 保存图像为base64编码，以便在前端显示
                        _, buffer = cv2.imencode('.jpg', img)
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # 检查是否有标注
                        annotations = Annotation.query.filter_by(data_file_id=data_file_id).all()
                        annotation_coordinates = ''
                        true_behavior = None
                        has_annotation = False
                        
                        # 查找该帧是否有标注
                        for ann in annotations:
                            if ann.timestamp is not None and int(ann.timestamp) == frame_index:
                                annotation_coordinates = ann.coordinates
                                true_behavior = ann.behavior
                                has_annotation = True
                                break
                        
                        # 保存帧预测信息
                        frame_predictions.append({
                            'frame_index': frame_index,
                            'behavior': predicted_behavior,
                            'image_data': f'data:image/jpeg;base64,{img_base64}',
                            'coordinates': annotation_coordinates,
                            'true_behavior': true_behavior
                        })
                        
                        # 更新准确率统计
                        if has_annotation:
                            # 统计该行为的总预测次数
                            behavior_accuracies[predicted_behavior]['total'] += 1
                            # 如果预测正确，统计正确次数
                            if predicted_behavior == true_behavior:
                                behavior_accuracies[predicted_behavior]['correct'] += 1
                                correct_predictions += 1
                            total_predictions += 1
                        
                        extracted_frames += 1
        
        # 计算各行为的准确率
        for behavior in behavior_accuracies:
            if behavior_accuracies[behavior]['total'] > 0:
                behavior_accuracies[behavior]['accuracy'] = behavior_accuracies[behavior]['correct'] / behavior_accuracies[behavior]['total']
        
        # 计算总体准确率
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # 保存评估结果
        evaluation = Evaluation(
            model_id=model_id,
            data_file_id=data_file_id,
            correct_predictions=correct_predictions,
            total_predictions=total_predictions,
            accuracy=overall_accuracy
        )
        db.session.add(evaluation)
        db.session.commit()
        
        return render_template('evaluate_result.html', 
                               model=model,
                               data_file=data_file,
                               frame_predictions=frame_predictions,
                               behavior_counts=behavior_counts,
                               behavior_accuracies=behavior_accuracies,
                               overall_accuracy=overall_accuracy,
                               behaviors=BEHAVIORS)
    
    # 获取所有可用的数据文件
    data_files = DataFile.query.all()
    return render_template('evaluate.html', model=model, data_files=data_files)
