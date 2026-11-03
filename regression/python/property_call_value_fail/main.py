class T:

    @property
    def path(self) -> str:
        return "s3://x"


def main() -> None:
    t = T()
    p = t.path()
    assert p != "zz"


main()
