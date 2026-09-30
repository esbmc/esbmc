# A class object used as a value is a well-formed char array, so encoding it
# during symex does not read past its members.
A = int


def sort2(h: list[int]) -> None:
    i = 0
    while i < len(h) - 1:
        if h[i + 1] < h[i]:
            t = h[i]
            h[i] = h[i + 1]
            h[i + 1] = t
        i = i + 1


xs = [3, 1]
sort2(xs)
assert xs[0] == 1
assert A == int
assert A != str
