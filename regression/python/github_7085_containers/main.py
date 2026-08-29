# The model container types carry a class tag too, so the no-__len__ refusal
# must not swallow their own len handling.

def main():
    l = [1, 2, 3]
    d = {1: 2, 3: 4}
    t = (1, 2)
    s = "abcd"
    assert len(l) == 3
    assert len(d) == 2
    assert len(t) == 2
    assert len(s) == 4


main()
