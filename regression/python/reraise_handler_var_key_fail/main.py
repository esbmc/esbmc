d = {1: 2}
try:
    x = d[3]
except LookupError as e:
    raise e
