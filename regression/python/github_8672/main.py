def f(xs, i):
    return "v %d" % xs[i]


def g(xs, i):
    try:
        return "v %d" % xs[i]
    except IndexError:
        return "none"


assert f([1, 2], 1) == "v 2"
assert g([1, 2], 5) == "none"
assert g([1, 2], 0) == "v 1"
d = {"a": 4}
assert "k=%d" % d["a"] == "k=4"
names = ["bob", "al"]
assert "%s and %s" % (names[0], names[1]) == "bob and al"
s = "abc"
assert "%s!" % s[1] == "b!"
assert f"{s[2]}{s[0]}" == "ca"
