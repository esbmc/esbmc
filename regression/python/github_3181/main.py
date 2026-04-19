class MyException(BaseException):
    def __init__(self, msg: str) -> None:
        self.msg = msg

def foo(x: int) -> None:
    if x == 42:
        raise MyException("bad number!")

import random

r = random.randint(41, 43)
try:
    foo(r)
except MyException as e:
    print("caught MyException exception")
