class First(Exception):
    pass


class Second(Exception):
    pass


try:
    raise Second
except First:
    b = 1
