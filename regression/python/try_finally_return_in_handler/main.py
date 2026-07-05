# A return inside an except handler escapes the try; the appended finally on
# the fall-through path would be bypassed, so this shape is refused rather than
# verified unsoundly. (Valid Python -- finally runs before the return under
# CPython, so it executes cleanly and exits 0 here.)
def f() -> int:
    try:
        raise ValueError()
    except ValueError:
        return 1
    finally:
        pass


print(f())
