def main():
    d: dict[str, int] = {"a": 1, "b": 2}
    count: int = 0
    for k, v in d.items():
        break
        count += 1
    assert count == 0

main()
