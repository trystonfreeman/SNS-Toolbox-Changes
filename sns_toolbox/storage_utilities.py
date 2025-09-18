"""
Utility functions for saving and loading compiled SNS networks.
"""
from sns_toolbox.backends import Backend,SNS_Numpy
import os
import json
import numpy as np
import torch
import pickle

def save(model: Backend, file_path: str) -> None:

    file_name, file_extension = os.path.splitext(file_path)

    # Determines whether file is saved as new .json or old .sns
    if file_extension == ".json":
        params = model.params

        # Converts ndarrays to lists that JSON module can handle
        for x,y in model.params:
            if isinstance(y,torch.Tensor):
                print('Torch Saving Not Implemented')
                return
            if isinstance(y,np.ndarray):
                params[x] = y.tolist()
        json.dump(params, open(file_path, 'w'))
    else:
        pickle.dump(model, open(file_path, 'wb'))


def load(file_path) -> Backend:
    file_name, file_extension = os.path.splitext(file_path)
    if file_extension == ".json":
        f = open(file_path)
        params = json.loads(f.read())
        for x,y in params.items():
            if type(y)==list:
                params[x] = np.array(y)
    else:
        model = pickle.load(open(file_path, 'rb'))
    return model