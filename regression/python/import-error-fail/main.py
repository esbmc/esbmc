try:
    import non_existent_module
except ImportError:
    caught = True
else:
    caught = False

assert not caught
