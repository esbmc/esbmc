# `raise e` re-raises the caught IndexError, not an Exception (#10242).
l = [1, 2]
try:
    x = l[5]
except Exception as e:
    raise e
