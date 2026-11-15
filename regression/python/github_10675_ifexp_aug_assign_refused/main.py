def func():
    return 1


flag = nondet_bool()
h = func if flag else 42
h += 1
