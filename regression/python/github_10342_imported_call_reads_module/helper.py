ENABLED = True


def call(h):
    if ENABLED:
        return h()
    return None
