# Exercises the --irep2-native-body IREP2-native code_break2t/code_continue2t
# dispatcher (W1-loc spike Phase C, esbmc/esbmc#4715): with_break()'s and
# with_continue()'s conditions and bodies are side-effect-free, so the whole
# while loop -- including the break/continue statements themselves, each an
# unconditional GOTO to the loop's break/continue target preceded by
# unwind_destructor_stack's DEAD instructions -- converts natively with no
# legacy round-trip. nested() additionally exercises continue in an inner
# loop and break in the same inner loop, both correctly resolving to the
# INNER loop's targets, not the outer one's. main-level code (calls +
# asserts) falls back to goto_convert_rec. Verdict and GOTO must match a run
# without the flag.


def with_break(n: int) -> int:
    i: int = 0
    while i < n:
        if i == 3:
            break
        i = i + 1
    return i


def with_continue(n: int) -> int:
    s: int = 0
    i: int = 0
    while i < n:
        i = i + 1
        if i == 2:
            continue
        s = s + i
    return s


def nested(n: int) -> int:
    total: int = 0
    i: int = 0
    while i < n:
        j: int = 0
        while j < n:
            if j == 1:
                j = j + 1
                continue
            if j == 3:
                break
            total = total + 1
            j = j + 1
        i = i + 1
    return total


r1: int = with_break(10)
r2: int = with_continue(5)
r3: int = nested(4)
assert r1 == 3
assert r2 == 13
assert r3 == 8
