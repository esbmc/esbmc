# A return inside an except handler escapes the try, so the finally is copied
# in front of it and runs before control leaves. Issue #7076: this shape used
# to be refused during conversion.
def f() -> int:
    try:
        raise ValueError()
    except ValueError:
        return 1
    finally:
        pass


assert f() == 1
