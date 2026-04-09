# HumanEval/6 debug P1.3
# Outer function has return annotation; internal function has no return annotation


def parse_nested_parens(paren_string: str) -> int:
    def parse_paren_group(s):
        return 0

    return parse_paren_group(paren_string)


if __name__ == "__main__":
    result = parse_nested_parens("()")
    assert result is not None
