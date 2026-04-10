try:
    import non_existent_module
    try:
        import another_non_existent_module
    except ImportError:
        inner_caught = True
except ImportError:
    outer_caught = True
    inner_caught = False

assert outer_caught
assert not inner_caught
