# 10461: a negative index past the start raises IndexError.
l = [0.5]
l.append(3)
raised = False
try:
    x = l[-3]
except IndexError:
    raised = True
assert raised
