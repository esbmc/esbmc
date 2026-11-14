import random


def text():
    return " ab "


def number():
    return 42


def strip_of(f):
    return f().strip()


f = text if random.random() < 0.5 else number
r = strip_of(f)
assert r == "ab"
