# The outer handler binds the re-raised object itself, extra fields included.
class MyError(IndexError):
    def __init__(self, code: int):
        self.code = code


caught = 0
try:
    try:
        raise MyError(7)
    except Exception as e:
        raise e
except MyError as m:
    caught = m.code
assert caught == 7
