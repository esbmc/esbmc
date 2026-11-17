try:
    code = "a = 1 // 0"
    exec(code)
except ZeroDivisionError:
    code = "a = 2"
    exec(code)
assert a == 2
