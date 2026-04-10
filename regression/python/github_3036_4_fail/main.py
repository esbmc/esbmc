def main() -> None:
    xs = [1, 2, 3, 4, 5, 6, 10]
    # Logic: Keep if (Even AND > 4) OR (exactly 1)
    # 1 -> Odd (Fail) OR 1==1 (True) -> Keep
    # 2 -> Even (True) AND > 4 (Fail) -> Fail
    # 6 -> Even (True) AND > 6 (True) -> Keep
    ys = [x for x in xs if (x % 2 == 0 and x > 4) or x == 1]
    
    assert ys == [1, 6, 11]

main()
