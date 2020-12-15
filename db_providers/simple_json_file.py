"""This module is for simply insert JSON objects to a file.

The hierarchy of URL - DB Name - Collection is just file inside folder inside a folder like this:

- URL (dir)
    - DB Name (dir)
        - Collection (JSON file)

"""
import os
import json
from pathlib import Path
import contextlib

from logger_process import INFO

TINYMONGO_FOLDER_NAME = 'simple_json_db'

CONNECT_STR = TINYMONGO_FOLDER_NAME


def init_collection(url: str, db_name: str, collection_name: str, log_func):
    """Create a collection inside a simple JSON DB folder"""
    Path(f"{url}/{db_name}").mkdir(parents=True, exist_ok=True)

    client = SimpleJsonFileClient(url)
    db = client[db_name]
    collection = db[collection_name]

    log_func(INFO, f"initialized simple JSON file collection {collection_name}")

    return collection


class SimpleJsonFileCollection:
    def __init__(self, url, db_name, collection_name, client):
        self.url = url
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = client

        # Open the file and append to client's open files
        file_path = f"{url}/{db_name}/{collection_name}.json"

        # Create if not exist
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                pass

        self.f = open(file_path, 'r+')
        self.client.files.append(self.f)

    def find(self, **kwargs):
        self.f.seek(0)

        return json.loads(self.f.read())

    def insert_one(self, doc):
        self.f.seek(0)

        try:
            docs = json.loads(self.f.read())
        except json.JSONDecodeError:
            docs = []

        docs.append(doc)

        self.f.seek(0)
        self.f.write(json.dumps(docs))


class SimpleJsonFileDB:
    def __init__(self, url, db_name, client):
        self.url = url
        self.db_name = db_name
        self.client = client

    def __getitem__(self, collection_name):
        return SimpleJsonFileCollection(self.url,
                                        self.db_name,
                                        collection_name,
                                        self.client)


class SimpleJsonFileClient:
    def __init__(self, folder_name):
        self.url = folder_name
        self.files = []

    def __getitem__(self, db_name):
        return SimpleJsonFileDB(self.url, db_name, self)

    def close(self):
        for f in self.files:
            f.close()


@contextlib.contextmanager
def get_client(folder_name: str):
    client = SimpleJsonFileClient(folder_name)
    yield client
    # Nothing to do for closing the client
    client.close()
