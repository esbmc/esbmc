# KNOWNBUG. The frontend is not flow-sensitive about a symbol's type: one
# symbol, one type. `s = "ab"` wants char[3] and the repeat yields a string of
# statically-unknown length, so the only type fitting both assignments is a
# bare pointer. The GOTO shows it -- with a fresh target the source stays an
# array and the model gets a static length, with a rebind it does not:
#
#     t = s * 3  ->  ASSIGN s={ 97, 98, 0 };      t=__python_str_repeat(&s[0], 3)
#     s = s * 3  ->  ASSIGN s=&{ 97, 98, 0 }[0];  s=__python_str_repeat(&s[0], 3)
#
# __python_str_repeat then has to walk the pointer, so its copy loop
# (src/c2goto/library/python/string.c:1593) has no static bound -- it reached
# iteration 9303 and was still climbing for this six-character result. Raising
# --unwind therefore cannot fix this test; the bound is gone, not too small.
# The fix is to mint a fresh symbol for the retyped `s`, as get_var_assign
# already does across the numeric<->string boundary (#4770/#4774).
# string-repeat-fresh-target pins the shape that does work.
s = "ab"
s *= 3
assert s == "ababab"
assert len(s) == 6
