# Exercises --python-irep2-adjust-only on string-element access over a pointer.
# A str variable is a char*; a bare string literal assigned to it is a
# constant_array that must decay to &array[0] (clang_c_adjust does this; the
# hop-off path needs python_adjust's array→pointer decay arm), and the element
# read s[i] over the resulting pointer must rewrite to *(s+i) (the index arm).
# Without both, symex crashes: "Unexpected index type in computer_pointer_offset"
# or an irep2_cast_error in fixup_renamed_type.
result = []
word = ""
for char in "a,":
    if char == ",":
        result.append(word)
    else:
        word += char
assert result[0] == "a"
