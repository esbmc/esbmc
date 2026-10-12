# A failing assert ends the program, so the ValueError the call raises after
# the following assertions is unreachable.
def check(i: int) -> None:
    if i < 0:
        raise ValueError("negative")


i = nondet_int()
assert i >= 0
assert i < 0 or i >= 0
check(i)
