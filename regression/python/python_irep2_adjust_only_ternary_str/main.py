# Exercises the --python-irep2-adjust-only array-to-pointer decay of a ternary
# branch. `"" if b else "foo"` builds a char*-typed ternary over two array
# literals, and migrate_expr coerces each branch with a plain typecast, giving
# `(char *){ 0 }`. clang decays it to `&{ 0 }[0]`; without the decay the SMT
# layer rejects the pointer-typed array constant with "Unexpected type in
# int/ptr typecast".
def pick(b: bool) -> str:
    s: str = "" if b else "foo"
    return s


assert len(pick(True)) == 0
assert len(pick(False)) == 3
