import numpy as np

def apple_or_orange(vector):
    """Takes a 3-input vector and returns whether the object is an
    apple or an orange.
    """
    weights = np.array([[0, 1, 1]])
    vector  = np.array([[dim] for dim in vector])
    sign    = sum(weights @ vector)
    print(weights @ vector)

    if (sign >= 0):
        return "Apple"
    else:
        return "Orange"

