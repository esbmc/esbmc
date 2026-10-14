import numpy as np

saved = None


def store():
    global saved
    a = np.array([[1, 2], [3, 4]])
    row = a[0]
    saved = row


store()
