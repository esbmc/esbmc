from numpy import zeros_like

base = [1]
out = zeros_like(base)

assert out[0] == 1
