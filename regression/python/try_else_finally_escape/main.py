# An escaping return/break/continue in the else clause of a try/finally would
# bypass the appended finally, so this shape is refused with a clean diagnostic
# rather than a silently-wrong verdict. (Valid Python; refused, not verified.)
def f():
    try:
        pass
    except ValueError:
        return 2
    else:
        return 3
    finally:
        pass
    return 0


assert f() == 3
