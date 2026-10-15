import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Mixer:
    def mix(self, k, w, scale):
        w[0] = w[0] * scale
        return w[0] + k

m = Mixer()
r = m.mix(1, v, 3)
assert r == 4
assert a[1] == 3
