try:
    import non_existent_module
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

assert not HAS_MODULE
