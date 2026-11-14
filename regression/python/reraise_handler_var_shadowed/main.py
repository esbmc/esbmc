# The inner handler rebinds e and unbinds it on exit, so `raise e` raises
# NameError, not the IndexError the outer handler caught.
l = [1, 2]
try:
    x = l[5]
except Exception as e:
    try:
        raise KeyError()
    except KeyError as e:
        pass
    raise e
