# Regression for a core dump in goto_convertt::do_function_call: calling a
# function-pointer parameter and nesting the result directly as an argument to
# another call (here list.append) produced a typecast callee that the
# goto-convert dispatcher aborted on. The typecast callee is now dereferenced
# through the pointer-to-code path. VERIFICATION FAILED (the appended value is 7, not 8).
def gen() -> int:
    return 7


def build(elem):
    l = []
    l.append(elem())
    return l


def main() -> None:
    xs = build(gen)
    assert xs[0] == 8


main()
