# repr() of a str literal quotes it (#7559).
def main() -> None:
    assert repr('a') == 'a'


main()
