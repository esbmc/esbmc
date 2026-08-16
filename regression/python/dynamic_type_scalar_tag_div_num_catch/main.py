cond = nondet_bool()
if cond:
    x = 10
else:
    x = "a"

if cond:
    caught = 0
    try:
        y = x / 0
    except ZeroDivisionError:
        caught = 1
    assert caught == 1
