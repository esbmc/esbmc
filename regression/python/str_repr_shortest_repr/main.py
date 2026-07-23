def main() -> None:
    # str()/repr()/f-string now fold a constant float to CPython's shortest
    # round-trip repr instead of refusing it: previously anything needing more
    # than 6 significant digits, or exponential notation, emitted a nondet
    # string. str() shares one renderer with "{}".format() now.
    assert str(1.23456789) == "1.23456789"
    assert str(3.14159265358979) == "3.14159265358979"
    assert str(0.30000000000000004) == "0.30000000000000004"
    assert str(123456789.5) == "123456789.5"
    assert repr(0.1234567) == "0.1234567"

    # Exponential ranges CPython uses outside [1e-4, 1e16).
    assert str(1e-05) == "1e-05"
    assert str(1e16) == "1e+16"
    assert str(1.5e20) == "1.5e+20"
    assert str(1e300) == "1e+300"

    # Fixed/exponential cut-over matches CPython on both sides.
    assert str(0.0001) == "0.0001"
    assert str(1e15) == "1000000000000000.0"

    # Short and whole-number values were already correct and stay so.
    assert str(2.5) == "2.5"
    assert str(0.1) == "0.1"
    assert str(1.0) == "1.0"
    assert str(1000000.0) == "1000000.0"

    # f-string interpolation routes through the same renderer.
    assert f"{1.23456789}" == "1.23456789"
    assert f"x={0.30000000000000004}" == "x=0.30000000000000004"


main()
