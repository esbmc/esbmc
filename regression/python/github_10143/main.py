# #10143: (xs and True) nested in another `and` is tested by its truth.
def bump(n: int) -> bool:
    return n > 0


xs = [1]
k = 0
while (xs and True) and k < 1:
    k += 1
assert k == 1

if (xs and True) and k < 2:
    k += 1
assert k == 2

while (xs and bump(1)) and k < 3:
    k += 1
assert k == 3
