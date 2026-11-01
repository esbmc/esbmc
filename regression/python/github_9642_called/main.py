class MyException(Exception):
    def __init__(self):
        self.val = 15


try:
    raise MyException()
except MyException as e:
    b = e.val
assert b == 15
