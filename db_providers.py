import pymongo
import tinymongo

from logger_process import INFO


def init_mongodb_collection(url: str, db_name: str, collection_name: str, log_func):
    """Create a collection inside a db in remote mongoDB server"""
    client = pymongo.MongoClient(url)
    db = client[db_name]
    collection = db[collection_name]

    log_func(INFO, f'initialized mongoDB collection {collection_name}')
    return collection


def init_tiny_mongo_collection(folder_name: str, db_name: str, collection_name: str, log_func):
    """Create a collection inside a tiny-mongo folder"""
    client = tinymongo.TinyMongoClient(folder_name)
    db = client[db_name]
    collection = db[collection_name]

    log_func(INFO, f'initialized tiny mongo collection {collection_name}')
    return collection
