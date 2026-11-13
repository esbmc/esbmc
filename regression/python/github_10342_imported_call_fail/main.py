from helper import call


def text():
    return "s"


def number():
    return 42


use_text = True
if use_text:
    f = text
else:
    f = number
r = call(f)
s = call(number)
assert r == "s"
assert s == 43
