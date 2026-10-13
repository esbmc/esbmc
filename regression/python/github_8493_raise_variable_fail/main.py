class E(Exception):
    pass


class F(Exception):
    pass


e = E()
try:
    raise e
except F:
    pass
