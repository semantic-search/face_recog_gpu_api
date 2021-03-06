import os
from dotenv import load_dotenv
load_dotenv()
SEND_TOPIC_FULL = "IMAGE_RESULTS"
SEND_TOPIC_TEXT = "TEXT"
KAFKA_HOSTNAME = os.getenv("KAFKA_HOSTNAME")
KAFKA_PORT = os.getenv("KAFKA_PORT")
REDIS_HOSTNAME = os.getenv("REDIS_HOSTNAME")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
ALLOWED_IMAGE_TYPES = ["jpg", "png"]
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_PASSWORD = os.getenv("KAFKA_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
DB = os.getenv('MONGO_DB')
PORT = os.getenv('MONGO_PORT')
MONGO_USER = os.getenv('MONGO_USER')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')

# CONTAINER NAME
RECEIVE_TOPIC = 'FACE_RECOG'

CONNECTED =  'CONNECTED'
DISCONNECTED = 'DISCONNECTED'

CORS_ORIGIN = os.getenv('CORS_ORIGIN')
embeddings = []

def add_to_embeddings(username, encoding):
    interm = {
        "name": username,
        "encoding": encoding
    }
    embeddings.append(interm)
