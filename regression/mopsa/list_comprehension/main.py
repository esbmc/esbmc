def f(x):
    return -x

l = [1, 10, 100, 1000]
m = [2*i for i in l]
n = [f(i) for i in l]
