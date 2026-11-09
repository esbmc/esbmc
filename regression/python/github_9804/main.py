import random

import helper


def func():
    return 42


if random.random() < 0.5:
    f = helper.func
else:
    f = func
assert f() == 42
