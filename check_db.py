from app import app, db
from models import Model
from sqlalchemy import inspect

with app.app_context():
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print('Database tables:', tables)
    
    # 检查是否存在model表
    if 'model' in tables:
        print('Model count:', Model.query.count())
    else:
        print('Model table does not exist!')
        # 尝试创建所有表
        db.create_all()
        print('Tables created!')
        print('Now model count:', Model.query.count())