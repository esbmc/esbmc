# A heterogeneous int/float list passed as a parameter and iterated with a
# for-loop must read each element at its true width, so an integral float
# element compares equal to its float literal and an int element promotes to
# float. Under the old list[int] fallback the loop variable was int: 4.0
# truncated to 4 so `gpa == 4.0` was false (esbmc/esbmc#5156).
def grade(xs):
    out = []
    for x in xs:
        if x == 4.0:
            out.append("A")
        elif x > 1.5:
            out.append("B")
        else:
            out.append("C")
    return out


if __name__ == "__main__":
    assert grade([4.0, 3, 1.7]) == ["A", "B", "B"]
