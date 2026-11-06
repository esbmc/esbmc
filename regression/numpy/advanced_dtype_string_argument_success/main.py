import numpy as np

def first(x):
    return x[0]


def count(x):
    return len(x)


def overwrite(x):
    x[0] = "zzz"


a = np.array(["ab", "c"])
assert first(a) == "ab"
assert count(a) == 2
assert "c" in a
assert "d" not in a
assert list(a)[1] == "c"
assert len(tuple(a)) == 2
overwrite(a)
assert a[0] == "zz"


def second(x):
    return x[1]


letters = np.array(["b", "a"])
assert second(letters) == "a"

def same(x):
    return x


held = same(a)
assert held[1] == "c"
text = np.array(["ab", "cd", "ef"])
window = text[0:2]
window[0] = "zz"
assert text[0] == "zz" and window[1] == "cd" and len(window) == 2
assert text[-2:][1] == "ef"
grid = np.array([["a", "b"], ["c", "d"]])
row = grid[0]
row[0] = "z"
assert grid[0][0] == "z"
copied = np.copy(text)
copied[1] = "qq"
assert text[1] == "cd"
sliced = text[0:2].copy()
sliced[0] = "yy"
assert text[0] == "zz"
matches = text == "cd"
assert matches[1] and not matches[0]
assert (text != "cd")[0]
picked = np.random.choice(text[1:3])
assert picked == "cd" or picked == "ef"
numbers = np.array([123, -4])
short_text = numbers.astype("U2")
long_text = numbers.astype(str)
assert short_text[0] == "12" and long_text[1] == "-4"
assert long_text.itemsize == 84
truth_values = np.array([True])
truth_text = truth_values.astype(str)
assert truth_text.itemsize == 20
one_character = text.astype("U1")
assert one_character[1] == "c"
raw = np.array([b"abcd", b"x"], dtype="S3")
raw[1] = b"zzzz"
assert raw[0] == b"abc" and raw[1] == b"zzz" and raw[0] != "abc"
assert len(raw) == 2

a_m1 = np.array(["ab", "c", ""])
assert a_m1[0] == "ab" and a_m1[1] == "c" and a_m1[2] == ""
assert len(a_m1[0]) == 2 and len(a_m1) == 3
a_m1[1] = "xyz"
assert a_m1[1] == "xy"

s_m1 = np.array(["ab", "c", "def"])
tail_m1 = s_m1[1:]
assert len(tail_m1) == 2 and tail_m1[0] == "c" and tail_m1[1] == "def"
assert tail_m1.itemsize == 12 and tail_m1.nbytes == 24
every_m1 = s_m1[::2]
assert every_m1[1] == "def"

a_m2=np.array(['ab','cd','ef'])
b_m2=a_m2[-2:]
assert b_m2[0]=='cd'
assert a_m2[1:][-1]=='ef'
x_m2=a_m2[0] if len(a_m2)==3 else a_m2[1]
assert x_m2=='ab'


a_m4=np.array([b'ab'],dtype='S2')
assert a_m4.size==1
assert a_m4.shape[0]==1
assert a_m4.ndim==1
