from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os 
import json
load_dotenv()
# Constants

uri = os.getenv('uri')

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# use a database named "myDatabase"
db = client.email

collection = db['email']

collection.delete_many({})

current_dir = os.path.dirname(__file__)
file_name = "cleaned_email_data_v4.json"
# data 폴더의 JSON 파일 경로를 생성
file_path = os.path.join(current_dir, '..', 'data', 'graph_rag', file_name)

with open(file_path,'r') as file:
    file_data = json.load(file)

collection.insert_many(file_data)