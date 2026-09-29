# A value popped from a list is an element, not a list: an unannotated
# parameter it is passed to must not be typed as a list (#4797).
def add(a, b):
    return a + b


stack = []
stack.append(3.0)
stack.append(5.0)
a = stack.pop()
b = stack.pop()
stack.append(add(b, a))
assert stack.pop() == 8.0
