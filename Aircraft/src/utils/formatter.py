import hashlib
import struct
import json
import numpy as np
import base64


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


def format_elapsed_time(elapsed_time):
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"


def hash_dict_keys(d: dict):
    sorted_dict = {k: d[k] for k in sorted(d)}
    keys_hash = ','.join(sorted_dict.keys())
    return keys_hash


def hash_dict_vals(d: dict):
    sorted_dict = {k: d[k] for k in sorted(d)}
    sorted_dict_vals = sorted_dict.values()
    sorted_dict_vals_str = list(map(str, sorted_dict_vals))
    vals_hash = ','.join(sorted_dict_vals_str)
    return vals_hash


def unhash_dict(keys_hash, vals_hash):
    keys = keys_hash.split(',')
    vals_str = vals_hash.split(',')
    vals = list(map(float, vals_str))
    return {k: v for k, v in zip(keys, vals)}


def hashed_vals_to_csv(key_hash: str, hashed_vals_set: set, filename: str):
    with open(filename, 'w') as f:
        # Write the keys hash as the first line:
        f.write(key_hash + '\n')
        for val_hash in hashed_vals_set:
            f.write(val_hash + '\n')


def csv_to_hashed_val_set(filename: str):
    hashed_val_set = set()
    with open(filename, 'r') as f:
        key_hash = f.readline().strip('\n')
        for line in f.readlines()[1:]:
            val_hash = line.strip('\n')
            hashed_val_set.add(val_hash)
    return key_hash, hashed_val_set
