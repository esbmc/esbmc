# !r and !a of None render "None" rather than a nondet string (#7559).
def main() -> None:
    assert f'{None!r}' == 'None'
    assert f'x{None!a}y' == 'xNoney'


main()
