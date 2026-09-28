def encode() -> str:
    import json
    return json.dumps([1, 2])


def main() -> None:
    assert len(encode()) > 0


main()
