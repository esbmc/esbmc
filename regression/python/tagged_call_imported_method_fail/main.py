import random

from helper import call


def text():
    return " ab "


def number():
    return 42


f = text if random.random() < 0.5 else number
assert call(f) == "ab"
