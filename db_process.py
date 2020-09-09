import pymongo
import multiprocessing
import functools


def init_mongo_collection(url: str, db_name: str, date_str: str):
    """Create a db inside remote mongoDB server which its name is the current time"""
    client = pymongo.MongoClient(url)
    db = client[db_name]

    return db[date_str]


def insert_to_db_loop(url: str, db_name: str, date_str: str, q: multiprocessing.Queue):
    """Insert results from queue to remote mongo database"""
    # First, get a DB connection
    collection = init_mongo_collection(url, db_name, date_str)

    # Send to remove DB instances data
    while True:
        instance_data = q.get()
        collection.insert_one(instance_data)


def insert_to_db_func(q: multiprocessing.Queue, instance_data):
    q.put(instance_data)


def start_db_process(url: str, db_name: str, date_str: str, q: multiprocessing.Queue):
    p = multiprocessing.Process(target=insert_to_db_loop,
                                args=(url, db_name, date_str, q))

    p.start()

    return p, functools.partial(insert_to_db_func, q)
