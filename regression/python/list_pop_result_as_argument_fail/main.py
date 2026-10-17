# 3.0 + 5.0 is 8.0 (#4797).
def add(a, b):
    return a + b


stack = []
stack.append(3.0)
stack.append(5.0)
a = stack.pop()
b = stack.pop()
stack.append(add(b, a))
assert stack.pop() == 9.0
