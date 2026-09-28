# !r of None renders "None" (#7559).
def main() -> None:
    assert f'{None!r}' == 'none'


main()
