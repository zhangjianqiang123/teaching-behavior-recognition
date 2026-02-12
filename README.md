# 课堂教学行为识别系统

一个基于Python Flask的课堂教学行为识别系统，用于实时分析课堂教学视频，识别和统计教师的教学行为。

## 功能特点

### 1. 数据管理
- 支持视频文件上传
- 自动提取视频帧（每3秒1帧）
- 数据文件管理

### 2. 标注功能
- 基于Canvas的可视化标注界面
- 支持绘制矩形框标注目标
- 支持多种教学行为标注
- 标注数据分页展示
- 缩略图预览和详情查看

### 3. 模型训练
- 基于SVM算法的行为分类模型
- 自动提取特征和训练分类器
- 训练过程可视化
- 模型准确率评估

### 4. 模型评估
- 支持对视频文件进行逐帧评估
- 行为统计和占比分析
- 准确率计算
- 帧级预测结果展示

### 5. 教学行为管理
- 支持增删改查教学行为类型
- 中英文标签支持
- 自定义行为类型

## 技术栈

### 后端
- Python 3.8+
- Flask - Web框架
- SQLAlchemy - ORM数据库框架
- SQLite - 轻量级数据库
- OpenCV - 视频处理和帧提取
- Scikit-learn - 机器学习算法
- Joblib - 模型保存和加载

### 前端
- HTML5 + CSS3 + JavaScript
- Bootstrap 5 - UI框架
- Canvas API - 标注功能
- Fetch API - 异步请求

## 项目结构

```
├── app.py                 # 应用入口文件
├── routes.py              # 路由和业务逻辑
├── models.py              # 数据库模型定义
├── templates/             # HTML模板文件
│   ├── base.html          # 基础模板
│   ├── index.html         # 首页
│   ├── upload.html        # 上传页面
│   ├── annotate.html      # 标注页面
│   ├── train.html         # 训练页面
│   ├── models.html        # 模型列表
│   ├── evaluate.html      # 评估页面
│   └── evaluate_result.html # 评估结果
├── static/                # 静态资源文件
│   ├── css/               # 样式文件
│   ├── js/                # JavaScript文件
│   ├── uploads/           # 上传的视频文件
│   ├── frames/            # 提取的视频帧
│   └── models/            # 训练好的模型
└── README.md              # 项目说明文档
```

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/teaching-behavior-recognition.git
cd teaching-behavior-recognition
```

### 2. 安装依赖

```bash
# 使用pip安装依赖
pip install -r requirements.txt

# 或使用conda
conda create -n teaching-behavior python=3.8
conda activate teaching-behavior
pip install -r requirements.txt
```

### 3. 创建必要的目录

```bash
mkdir -p static/uploads static/frames static/models models
```

### 4. 启动应用

```bash
python app.py
```

应用将在 http://127.0.0.1:5000 启动

## 使用说明

### 1. 上传视频

- 进入首页，点击"上传视频"按钮
- 选择要上传的视频文件（支持MP4, AVI, MOV等格式）
- 等待视频上传完成，系统会自动提取帧（每3秒1帧）

### 2. 标注数据

- 上传完成后，点击"标注"按钮进入标注页面
- 选择教学行为类型
- 在视频帧上绘制矩形框标注目标
- 点击"保存标注"按钮保存标注信息
- 可以通过分页查看已标注的信息

### 3. 训练模型

- 进入"模型训练"页面
- 系统会显示已标注的数据统计
- 点击"开始训练模型"按钮开始训练
- 等待训练完成，系统会显示训练结果

### 4. 评估模型

- 进入"模型评估"页面
- 选择要评估的模型和数据文件
- 点击"开始评估"按钮开始评估
- 等待评估完成，查看评估结果

### 5. 管理教学行为

- 进入"教学行为管理"页面
- 可以添加、编辑、删除教学行为类型
- 支持中英文标签

## 训练流程

1. **数据准备**：上传视频，系统自动提取帧
2. **标注数据**：为每个帧标注教学行为和目标位置
3. **特征提取**：从标注的帧中提取特征
4. **模型训练**：使用SVM算法训练分类模型
5. **模型评估**：评估模型准确率
6. **应用模型**：使用训练好的模型分析新视频

## 评估流程

1. **选择模型**：选择要评估的训练模型
2. **选择数据**：选择要评估的数据文件
3. **帧处理**：系统自动处理视频帧
4. **行为识别**：使用模型识别每帧的教学行为
5. **结果统计**：统计各行为出现次数和准确率
6. **结果展示**：显示评估结果和帧级预测详情

## 许可证

本项目采用 MIT 许可证，详情请参阅 [LICENSE](LICENSE) 文件。

## 贡献指南

欢迎各位开发者贡献代码和提出建议！

1. Fork 本仓库
2. 创建新的分支 (`git checkout -b feature/feature-name`)
3. 提交你的修改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/feature-name`)
5. 提交 Pull Request

## 联系方式

如果有任何问题或建议，欢迎通过以下方式联系：

- 项目地址：https://github.com/your-username/teaching-behavior-recognition
- Issues：https://github.com/your-username/teaching-behavior-recognition/issues

## 致谢

感谢所有为本项目做出贡献的开发者和支持者！

---

**课堂教学行为识别系统** - 让课堂教学分析更智能、更高效！