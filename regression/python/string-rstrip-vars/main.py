def main() -> None:
    s1 = "  alpha  "
    s2 = "\tbravo\n"
    s3 = "charlie"
    s4 = "delta   "
    s5 = "   echo"

    assert s1.rstrip() == "  alpha"
    assert s2.rstrip() == "\tbravo"
    assert s3.rstrip() == "charlie"
    assert s4.rstrip() == "delta"
    assert s5.rstrip() == "   echo"

    s1 = "  "
    s2 = ""
    s3 = "foxtrot \t"

    assert s1.rstrip() == ""
    assert s2.rstrip() == ""
    assert s3.rstrip() == "foxtrot"


main()
