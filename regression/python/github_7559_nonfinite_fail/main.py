# A non-finite float literal's !r is not "None" (#7559).
def main() -> None:
    assert f'{1e999!r}' == 'None'


main()
