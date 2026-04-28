def use_import(x):
    from other import inc

    return inc(x)


assert use_import(1) == 2
