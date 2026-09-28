# repr() of a str literal handed the literal's address to __python_str_repr,
# whose loops then never terminated (#7559).
def main() -> None:
    assert repr('a') == "'a'"
    assert f'{"it\'s"!r}' == '"it\'s"'


main()
