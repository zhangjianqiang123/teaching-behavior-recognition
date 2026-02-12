from app import app
from routes import train_model

# 直接调用train_model函数进行测试
if __name__ == '__main__':
    with app.app_context():
        train_model()