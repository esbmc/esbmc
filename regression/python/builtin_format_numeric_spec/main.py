def main() -> None:
    # The builtin format(value, spec) previously rejected any spec beyond a
    # bare presentation type ("08x", ".2f", ",", ">5" all errored). It now
    # folds the format-spec mini-language for numeric values, matching CPython.

    # Integer: width, zero-pad, and the sign/base/alternate-form options.
    assert format(5, "03d") == "005"
    assert format(42, "5") == "   42"
    assert format(42, "<5") == "42   "
    assert format(42, "^5") == " 42  "
    assert format(255, "08x") == "000000ff"
    assert format(255, "#x") == "0xff"
    assert format(255, "#08X") == "0X0000FF"
    assert format(5, "+") == "+5"
    assert format(-5, "05") == "-0005"
    assert format(-5, "=6") == "-    5"
    assert format(5, "*>4") == "***5"

    # The '0' flag sets a '0' fill but keeps an explicit alignment: ">05d" is
    # "00005", not "    5". An explicit fill char still wins over the '0' flag.
    assert format(5, ">05d") == "00005"
    assert format(5, "<05d") == "50000"
    assert format(5, "^05d") == "00500"
    assert format(5, "*>05d") == "****5"

    # Integer grouping (thousands separators).
    assert format(1000000, ",") == "1,000,000"
    assert format(1234, "+,") == "+1,234"

    # Float: precision, sign, width, zero-pad and the presentation types.
    assert format(3.14159, ".2f") == "3.14"
    assert format(3.14159, "8.2f") == "    3.14"
    assert format(3.14159, "08.2f") == "00003.14"
    assert format(-3.14159, "08.2f") == "-0003.14"
    assert format(3.14159, "+.2f") == "+3.14"
    assert format(1000.0, "e") == "1.000000e+03"
    assert format(1000.0, ".2E") == "1.00E+03"
    assert format(0.5, "%") == "50.000000%"
    assert format(0.5, ".1%") == "50.0%"


main()
