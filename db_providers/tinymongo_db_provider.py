import tinymongo
import contextlib

from logger_process import INFO

TINYMONGO_FOLDER_NAME = 'results_db'

CONNECT_STR=TINYMONGO_FOLDER_NAME

def init_collection(folder_name: str, db_name: str, collection_name: str, log_func):
    """Create a collection inside a tiny-mongo folder"""
    client = tinymongo.TinyMongoClient(folder_name)
    db = client[db_name]
    collection = db[collection_name]

    log_func(INFO, f'initialized tiny mongo collection {collection_name}')
    return collection


@contextlib.contextmanager
def get_client(folder_name: str):
    client = tinymongo.TinyMongoClient(folder_name)
    yield client
    client.close()
