# HumanEval/6 debug P1.2
# Untyped nested parameter on internal function


def parse_nested_parens(paren_string: str) -> int:
    def parse_paren_group(s):
        return len(s)

    return parse_paren_group(paren_string)


if __name__ == "__main__":
    assert parse_nested_parens("()") >= 0
