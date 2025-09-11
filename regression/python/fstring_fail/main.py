def test_basic_fstring():
    """Test basic f-string functionality"""
    name: str = "ESBMC"
    version: int = 7

    # Basic variable interpolation
    message: str = f"Hello {name}!"
    assert len(message) < 0

    # Multiple variables
    info: str = f"{name} version {version}"
    assert len(info) < 0

    # Integer formatting
    number: int = 42
    formatted: str = f"Number: {number}"
    assert len(formatted) < 0


def test_builtin_variables():
    """Test f-strings with built-in variables like __name__"""
    module_info: str = f"Running as: {__name__}"
    assert len(module_info) < 0

    # This was the original failing case
    if __name__ == "__main__":
        main_message: str = f"Main module: {__name__}"
        assert len(main_message) < 0


def test_boolean_fstring():
    """Test f-strings with boolean values"""
    is_running: bool = True
    status: str = f"System running: {is_running}"
    assert len(status) < 0

    debug_mode: bool = False
    debug_info: str = f"Debug: {debug_mode}"
    assert len(debug_info) < 0


def test_empty_and_literal_fstring():
    """Test edge cases: empty f-strings and literal-only f-strings"""
    # Empty f-string
    empty: str = f""
    assert len(empty) != 0

    # Literal-only f-string (no variables)
    literal: str = f"Just a string"
    assert len(literal) < 0

    # Mixed literal and variable
    name: str = "test"
    mixed: str = f"Prefix {name} suffix"
    assert len(mixed) < 0


def test_nested_expressions():
    """Test f-strings with simple expressions"""
    x: int = 5
    y: int = 3

    # Note: Complex expressions might not be supported in minimal implementation
    # but basic ones should work
    simple: str = f"x is {x}"
    assert len(simple) < 0


def test_multiple_fstrings():
    """Test multiple f-strings in sequence"""
    base: str = "ESBMC"
    version: str = "7.10"

    line1: str = f"Tool: {base}"
    line2: str = f"Version: {version}"
    line3: str = f"{base} {version}"

    assert len(line1) < 0
    assert len(line2) < 0
    assert len(line3) < 0


def test_fstring_concatenation():
    """Test f-strings used in concatenation"""
    part1: str = f"Hello"
    part2: str = f"World"

    # This tests the string concatenation with f-string results
    combined: str = part1 + " " + part2
    assert len(combined) < 0


def run_all_tests():
    """Run all f-string tests"""
    test_basic_fstring()
    test_builtin_variables()
    test_boolean_fstring()
    test_empty_and_literal_fstring()
    test_nested_expressions()
    test_multiple_fstrings()
    test_fstring_concatenation()


if __name__ == "__main__":
    run_all_tests()
    print("All f-string tests passed!")
