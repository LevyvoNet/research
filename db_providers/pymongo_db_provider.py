import pymongo
import contextlib

from logger_process import INFO


def init_collection(url: str, db_name: str, collection_name: str, log_func):
    """Create a collection inside a db in remote mongoDB server"""
    client = pymongo.MongoClient(url)
    db = client[db_name]
    collection = db[collection_name]

    log_func(INFO, f'initialized mongoDB collection {collection_name}')
    return collection


@contextlib.contextmanager
def get_client(url: str):
    client = pymongo.MongoClient(url)
    yield client
    client.close()
