# defined by --nondet-str-length (default = 16). The last character is always
# the null terminator. The visible string length is therefore <= max_len - 1.


def test_length_is_non_negative():
    s = nondet_str()
    assert len(s) >= 0  # Always true


test_length_is_non_negative()


def test_length_respects_upper_bound():
    s = nondet_str()
    # If max_str_length is N, usable chars are in [0, N-1)
    assert len(s) < 16  # default max_len = 16 → max visible length = 15


test_length_respects_upper_bound()


def test_function_argument_pass_through():

    def check_string(x: str) -> bool:
        return len(x) >= 0

    s = nondet_str()
    assert check_string(s)


test_function_argument_pass_through()


def test_non_empty_string_comparison():
    s = nondet_str()
    if len(s) > 0:
        assert s != ""  # Non-empty string cannot equal empty string


test_non_empty_string_comparison()


def test_assume_empty_string():
    s = nondet_str()
    __ESBMC_assume(len(s) == 0)
    assert s == ""  # Empty string should be represented as ""


test_assume_empty_string()
