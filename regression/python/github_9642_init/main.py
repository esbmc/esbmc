# Issue #9642: raise C runs C's own __init__.
class MyException(Exception):
    def __init__(self):
        self.code = 7

try:
    raise MyException
except MyException as e:
    b = e.code
assert b == 7
