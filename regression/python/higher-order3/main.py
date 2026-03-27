def make_multiplier(k):
    def mul(x):
        return x * k
    return mul

times3 = make_multiplier(3)
times3(4)   # 12
