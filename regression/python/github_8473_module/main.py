import helper
from helper import f

assert helper.f is f
assert f is helper.f
assert helper.g is not f
