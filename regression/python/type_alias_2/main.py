import typing as t

UserId = t.NewType('UserId', int)

def main() -> None:
    uid: UserId = UserId(42)
    assert uid == 42

main()
