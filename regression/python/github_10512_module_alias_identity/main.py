import helper

c = True
if c:
    f = helper.alias
else:
    f = helper.func
r = f()
assert f is helper.func
