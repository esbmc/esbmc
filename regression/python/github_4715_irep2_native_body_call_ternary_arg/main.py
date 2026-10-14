# A ternary argument is not a side effect, so the native call arms passed it
# straight to do_function_call -- whose remove_sideeffects the arms' comment
# called "a no-op we can skip issuing". Under --validate-violation-witness that
# is false: remove_sideeffects lowers `c ? a : b` to DECL/IF/GOTO, and the
# operands the arms hand it were never stamped, so the lowered instructions came
# out unlocated. Both shapes below reach the arms that were wrong: an assignment
# whose right-hand side is the call, and the call as a bare statement.
g_seen: int = 0


def g(v: int) -> int:
    return v


def f(c: bool, a: int, b: int) -> int:
    r: int = g(a if c else b)
    g(b if c else a)
    return r


assert f(True, 1, 2) == 1
