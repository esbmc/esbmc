def main() -> None:
    # Right-align with default fill (space).
    s1 = f"{42:>10}"
    assert len(s1) == 10
    assert s1[8] == "4" and s1[9] == "2"
    assert s1[0] == " "

    # Left-align.
    s2 = f"{42:<5}"
    assert len(s2) == 5
    assert s2[0] == "4" and s2[1] == "2"
    assert s2[4] == " "

    # Center align.
    s3 = f"{7:^5}"
    assert len(s3) == 5
    assert s3[0] == " " and s3[2] == "7" and s3[4] == " "

    # Custom fill.
    s4 = f"{42:*>6}"
    assert len(s4) == 6
    assert s4[0] == "*" and s4[3] == "*" and s4[4] == "4" and s4[5] == "2"

    # Width <= content length: no padding.
    s5 = f"{12345:>3}"
    assert len(s5) == 5
main()
