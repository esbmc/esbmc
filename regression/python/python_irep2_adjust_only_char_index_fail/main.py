# Negative counterpart to python_irep2_adjust_only_char_index: same hop-off path
# (--python-irep2-adjust-only, clang_cpp_adjust skipped), but the assertion is
# false, so the sole-adjuster path must still reach the solver and report the
# violation rather than crash. Proves the hop-off produces a real FAILED verdict,
# not a spurious success or an abort.
def test_chr_ord() -> bool:
    char_code = ord('A')
    back_to_char = chr(char_code)
    return back_to_char == 'A' and char_code == 65


assert test_chr_ord() == False
