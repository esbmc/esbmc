# def f(cond):
#     if cond:
#         return 1
#     else:
#         return "a"

# x = f(nondet_bool()) 
# assert x == 1 or x == "a"   

# def f(cond1, cond2):
#     if cond1:
#         return 1
#     elif cond2:
#         return 2
#     else:
#         return 'a'

def f(n):
    c = [(1, 2), (3, 4)]
    k = lambda t: t[0]
    assert k(c[0]) == 1

f(3)
    