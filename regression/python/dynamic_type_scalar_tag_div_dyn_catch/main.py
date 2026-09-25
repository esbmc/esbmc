cond = nondet_bool()
if cond:
    x = 10
    y = 0
else:
    x = "a"
    y = "b"

if cond:
    caught = 0
    try:
        z = x / y
    except ZeroDivisionError:
        caught = 1
    assert caught == 1
