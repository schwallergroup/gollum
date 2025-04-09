from argparse import Namespace
from collections.abc import MutableMapping
import importlib


def flatten_namespace(ns, parent_key="", separator="."):
    items = {}
    for k, v in ns.__dict__.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, Namespace):
            items.update(flatten_namespace(v, new_key, separator=separator))
        elif isinstance(v, dict):
            items.update(
                flatten_namespace(Namespace(**v), new_key, separator=separator)
            )
        else:
            items[new_key] = v
    return items


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_to_nested_dict(flat_dict):
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(".")
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        if value == "True" or value == "False":
            d[keys[-1]] = value == "True"
    return nested_dict


import copy


def instantiate_class(input_dict, *args, **kwargs):
    class_path = input_dict["class_path"]
    init_args = copy.deepcopy(input_dict.get("init_args", {}))

    # Convert init_args to dictionary if it is a Namespace
    if isinstance(init_args, Namespace):
        init_args = vars(init_args)

    init_args.update(kwargs)  # merge extra arguments into init_args

    # Iterate over init_args, checking if any values are themselves classes to be instantiated
    for arg_name, arg_value in init_args.items():
        if isinstance(arg_value, dict) and "class_path" in arg_value:
            init_args[arg_name] = instantiate_class(arg_value)

    module_name, class_name = class_path.rsplit(".", 1)
    MyClass = getattr(importlib.import_module(module_name), class_name)
    instance = MyClass(*args, **init_args)  # passing extra arguments to the class

    return instance
