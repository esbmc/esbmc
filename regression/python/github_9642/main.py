class MyException(Exception):
    pass


try:
    raise MyException
except MyException:
    b = 1
assert b == 1
