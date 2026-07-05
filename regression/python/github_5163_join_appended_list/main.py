# str.join() over a list built with append() must reflect the appended
# elements, not the declaration initializer. Before #5163, handle_str_join
# folded the variable's initializer ([]) and ignored later .append(...) calls,
# so " ".join(new_lst) returned "" instead of the real joined string.

new_lst = []
new_lst.append("is")
assert " ".join(new_lst) == "is"

# Multiple appends, joined in source order with a separator.
parts = []
parts.append("go")
parts.append("for")
assert " ".join(parts) == "go for"

# A never-mutated list literal must still fold correctly (no regression).
lit = ["a", "b"]
assert " ".join(lit) == "a b"
