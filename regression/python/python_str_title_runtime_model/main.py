# Runtime operational model for str.title.
#
# Before this change, handle_string_title required a compile-time
# constant receiver; non-constant receivers fell back to
# build_nondet_string_fallback (sound but unconstrained -- subsequent
# comparisons could never hold exactly).
#
# __python_str_title(const char *) -- added to
# src/c2goto/library/python/string.c -- is a bounded ASCII transform
# that uppercases the first letter of each "word" and lowercases
# every subsequent letter. A word starts on any letter immediately
# preceded by a non-letter (or by the start of the string). Non-
# letter characters pass through unchanged at their positions.
# 255-char cap on the receiver, same as the other case-transform
# models (lower / upper / swapcase / capitalize).
#
# Note: this model matches CPython's `str.title()` semantics, which
# treat ASCII digits as word breaks (so `"123abc"` titles to
# `"123Abc"`). The Python interpreter's `eval` path for constant
# receivers has a separate, slightly different rule (digits don't
# break words); that pre-existing divergence is out of scope here.
#
# The call sites below go through an `if do_it:` wrapper so consteval
# bails (control flow in the body) and the runtime model actually
# fires, rather than the call being folded at conversion time.


def t(s: str, do_it: bool) -> str:
    if do_it:
        return s.title()
    return s


if __name__ == "__main__":
    assert t("hello world", True) == "Hello World"
    assert t("HELLO WORLD", True) == "Hello World"
    assert t("hello-world", True) == "Hello-World"
    assert t("123abc def", True) == "123Abc Def"
    assert t("a b c", True) == "A B C"
    assert t("", True) == ""
    assert t("helloWorld", True) == "Helloworld"
