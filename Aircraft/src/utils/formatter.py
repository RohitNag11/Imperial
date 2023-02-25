import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def humanise_str(string):
    return string.replace('_', ' ').title()


def to_dict(obj):
    """
    Takes an object and recursively returns a dictionary containing keys and values as base classes.
    """
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(e) for e in obj]
    elif hasattr(obj, '__dict__'):
        return to_dict(obj.__dict__)
    else:
        return obj


def to_json(obj):
    return json.dumps(to_dict(obj), indent=4, cls=NumpyEncoder)


def save_obj_to_file(obj, filename):
    """
    Saves a JSON string to a text file.
    """
    json_str = to_json(obj)
    with open(filename, 'w') as file:
        json.dump(json.loads(json_str), file, indent=4)
