def main():
    d: dict[str, int] = {"a": 1}
    for k, v in d.items():
        k = "modified"
        v = 10
    assert True

main()
