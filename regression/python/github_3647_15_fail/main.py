def main():
    d: dict[str, int] = {"a": 1}
    k: int = 100
    for k, v in d.items():
        assert k == "a"
    assert k == 100  # outer scope unchanged

main()
