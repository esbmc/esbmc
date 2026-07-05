# An unannotated function whose every `return` is nested inside a branch and
# yields a runtime (opaque) list — here str.split() on a parameter receiver.
# Return-type inference previously scanned only the top-level statements, found
# no RETURN (both are inside the if/else), left the return type empty -> void,
# and remove_returns stripped the value, so the caller read a nondet list and
# the comparison reduced to NONDET. Recursive inference now recovers the list
# return type from the nested RETURNs.


def f(txt: str, c: bool):
    if c:
        return txt.split()
    else:
        return txt.split(',')


assert f("Hello world!", True) == ["Hello", "world!"]
