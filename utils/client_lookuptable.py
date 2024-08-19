import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import json
# Create a new client and connect to the server
load_dotenv()
# Constants

uri = os.getenv('uri')

client = MongoClient(uri, server_api=ServerApi('1'))
# use a database named "myDatabase"
db = client.email

email_collection = db['email']
