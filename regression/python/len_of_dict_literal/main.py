# len() of a dict *literal* reports the wrong size; binding it to a name first
# works, as do list and tuple literals. Predates the dunder-dispatch fix and is
# untouched by it.

assert len([1, 2, 3]) == 3      # list literal: fine
assert len((1, 2)) == 2         # tuple literal: fine
d = {"a": 1, "b": 2}
assert len(d) == 2              # named dict: fine
assert len({"a": 1, "b": 2}) == 2   # dict literal: wrong size
