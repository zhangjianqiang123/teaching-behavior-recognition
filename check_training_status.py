from app import app
from routes import TRAINING_STATUS, TRAINING_LOCK

print('Training status:', TRAINING_STATUS)
print('Training lock:', TRAINING_LOCK.locked())