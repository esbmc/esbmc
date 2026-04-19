try:
    import non_existent_module
except ImportError as e:
    caught = True

assert caught
