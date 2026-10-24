# A lambda default and a lambda argument return different types (#8970).
def call(h=lambda: "a"):
    return h()


assert isinstance(call(lambda: 1), int)
