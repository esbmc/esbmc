# A non-integral float element of a heterogeneous int/float list parameter must
# survive iteration. Under the old list[int] fallback the loop variable was int,
# so 1.7 truncated to 1 and the band check failed (esbmc/esbmc#5156).
def first_band(xs):
    for x in xs:
        return x > 1.3 and x < 1.9
    return False


if __name__ == "__main__":
    assert first_band([1.7, 2]) == True
