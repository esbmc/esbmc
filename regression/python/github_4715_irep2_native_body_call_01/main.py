# Exercises the --irep2-native-body IREP2-native code_function_call2t
# dispatcher (W1-loc spike Phase C, esbmc/esbmc#4715): bump()/reset()'s
# global-variable bodies are side-effect-free assignments, and each call to
# them from bump_twice()/run() is a bare "foo();" statement (return value
# unused, plain named callee, side-effect-free arguments), so goto_convert
# consumes each as a single FUNCTION_CALL with no legacy round-trip. Since
# the return value is never used, do_function_call's temp-symbol machinery
# (whose naming depends on a shared, order-sensitive counter) is never
# entered by this kind at all. main-level code (calls + asserts) falls back
# to goto_convert_rec. Verdict and GOTO must match a run without the flag.

g: int = 0


def bump(x: int) -> None:
    global g
    g = g + x


def reset() -> None:
    global g
    g = 0


def bump_twice(x: int, y: int) -> None:
    bump(x)
    bump(y)


def run() -> int:
    reset()
    bump_twice(2, 3)
    return g


r: int = run()
assert r == 5
