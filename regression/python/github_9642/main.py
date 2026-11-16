# Issue #9642: raise C raises C().
class MyException(Exception):
    pass

try:
    raise MyException
except MyException:
    b = 1
assert b == 1
