# Exercises the --python-irep2-adjust deref-result-type arm (docs/scope-v1k-adjuster
# round-4): a char-array element access chr(c)[0] produces a dereference whose
# result type the converter leaves empty. Under the flag, clang_cpp_adjust runs
# first and resolves it, so the pass stays inert and the verdict matches the
# default path; the arm's own resolution is what the hop-off flip census
# exercises (it turns this test's symbolic_type_excp crash into a verdict).
def test_chr_ord() -> bool:
    char_code = ord('A')
    back_to_char = chr(char_code)
    return back_to_char == 'A' and char_code == 65


assert test_chr_ord() == True
