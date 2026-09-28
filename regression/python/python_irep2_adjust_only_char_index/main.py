# Exercises --python-irep2-adjust-only: the IREP2-native Python adjuster runs as
# the SOLE adjuster (clang_cpp_adjust is skipped). The `back_to_char == 'A'`
# comparison lowers to a char-array element access whose dereference result type
# the converter leaves empty; with no legacy pass to resolve it, python_adjust's
# own deref-result arm (#6340) must resolve it, or symex aborts with
# type2t::symbolic_type_excp. A correct verdict here proves the hop-off path is
# self-sufficient on this shape.
def test_chr_ord() -> bool:
    char_code = ord('A')
    back_to_char = chr(char_code)
    return back_to_char == 'A' and char_code == 65


assert test_chr_ord() == True
