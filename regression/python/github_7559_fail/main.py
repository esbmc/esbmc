# repr('a') is "'a'", quotes included, so this does not hold (#7559).
def main() -> None:
    assert f'{"a"!r}' == 'a'


main()
