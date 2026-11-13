# 10461: a negative index past the start raises IndexError.
l = [0.5]
l.append(3)
x = l[-3]
