# !r and !a render repr(): an int's digits, a string in quotes. These were
# lowered to a nondet string, so any assertion about them failed (#7559).
def main() -> None:
    assert f'{"a"}' == 'a'
    assert f'{"a"!s}' == 'a'
    assert f'{"a"!r}' == "'a'"
    assert f'{"a"!a}' == "'a'"
    assert f'{1!r}' == '1'
    assert f'{42!a}' == '42'


main()
