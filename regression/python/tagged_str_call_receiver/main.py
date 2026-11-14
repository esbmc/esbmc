import random


def text():
    return " ab "


def number():
    return 42


def strip_of(f):
    return f().strip()


f = text
if random.random() > 2:
    f = number
assert strip_of(f) == "ab"
