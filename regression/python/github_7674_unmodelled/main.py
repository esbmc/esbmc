import json


def main() -> None:
    data: str = json.dumps([1, 2])
    assert len(data) > 0


main()
