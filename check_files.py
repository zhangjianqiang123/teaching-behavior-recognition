from app import app, db
from models import DataFile

with app.app_context():
    files = DataFile.query.all()
    print('文件数量:', len(files))
    for f in files:
        print(f'id: {f.id}, name: {f.filename}, type: {f.file_type}, status: {f.status}, path: {f.filepath}')
