import json


def load_json(path):
    with open(path) as json_file:
        config = json.load(json_file)
    return config
