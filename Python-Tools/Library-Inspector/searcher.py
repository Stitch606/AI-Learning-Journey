import inspect
import importlib


def search(module_name):
    module = importlib.import_module(module_name)

    result = {
        "m": [],
        "f": [],
        "c": [],
        "b": []
    }

    for name in dir(module):
        if name[0] == "_" or name[-1] == "_":
            continue

        obj = getattr(module, name)

        if inspect.ismodule(obj):
            result["m"].append(name)

        elif inspect.isfunction(obj):
            result["f"].append(name)

        elif inspect.isclass(obj):
            result["c"].append(name)

        
        elif inspect.isbuiltin(obj):
            result["b"].append(name)
            
    return result
