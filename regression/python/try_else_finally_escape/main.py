# An escaping return in the else clause of a try/finally gets the finally
# copied in front of it, so it no longer bypasses it. The else clause runs on
# the no-exception path, so f() returns 3 as under CPython (issue #7076).
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
