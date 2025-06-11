"""
Utility functions for saving and loading compiled SNS networks.
"""
from sns_toolbox.backends import Backend,SNS_Numpy
import json

# Allows Storage Utilities to work on micropython installations with ulab
try:
    from ulab import numpy as np
    micropython = True
except ImportError:
    import numpy as np
    import torch
    import pickle
    micropython = False
def save(model: Backend, filename: str,params: dict = None) -> None:

    # Required if using model for micropython
    if params != None:
        print("Saved model will only work on micro-SNS!")
        # Converts ndarrays to lists that JSON module can handle
        for x,y in params:
            if isinstance(y,torch.Tensor):
                print('Pytorch not compatable with micro-SNS!')
                return
            if isinstance(y,np.ndarray):
                params[x] = y.tolist()
        json.dump(params, open(filename, 'wb'))
    else:
        pickle.dump(model, open(filename,'wb'))

def load(filename) -> Backend:
    if not micropython:
        model = pickle.load(open(filename, 'rb'))
    else:
        f = open(filename)
        params = json.loads(f.read())
        for x,y in params.items():
            if type(y)==list:
                params[x] = np.array(y)
    return params