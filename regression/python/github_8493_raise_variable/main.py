class E(Exception):
    pass


def raise_local() -> None:
    e = E()
    raise e


e = E()
ok = False
try:
    raise e
except E:
    ok = True
assert ok

ok = False
try:
    raise_local()
except E:
    ok = True
assert ok

v = ValueError("bad")
ok = False
try:
    raise v
except ValueError:
    ok = True
assert ok

ok = False
try:
    try:
        raise E("inner")
    except E as err:
        raise err
except E:
    ok = True
assert ok
