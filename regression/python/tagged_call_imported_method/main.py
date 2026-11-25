import random

from helper import call


def text():
    return " ab "


def number():
    return 42


f = text
if random.random() > 2:
    f = number
assert call(f) == "ab"
