n = 1000
l = list(range(n))
r = []
k = 0
while k < 20:
    m = l[:]
    m[0] = k
    r.append(m)
    k += 1
