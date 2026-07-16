# Exercises the --irep2-native-body IREP2-native code_while2t dispatcher
# (W1-loc spike Phase C, esbmc/esbmc#4715): sum_to()'s loop condition and body
# are both side-effect-free (Python assignment is a genuine statement, unlike
# C's expression-statement-wrapped assign, so the body's code_assign2t nodes
# convert natively too), and the loop contains no break/continue, so
# goto_convert consumes the whole while loop natively (v: if(!c) goto z;
# x: P; y: goto v; z: ;) with no legacy round-trip. nested() additionally
# exercises a while nested inside another while's body. main-level code
# (calls + asserts) falls back to goto_convert_rec. Verdict and GOTO must
# match a run without the flag.


def sum_to(n: int) -> int:
    s: int = 0
    i: int = 0
    while i < n:
        s = s + i
        i = i + 1
    return s


def nested(n: int) -> int:
    i: int = 0
    total: int = 0
    while i < n:
        j: int = 0
        while j < n:
            total = total + 1
            j = j + 1
        i = i + 1
    return total


r1: int = sum_to(5)
r2: int = nested(3)
assert r1 == 10
assert r2 == 9
