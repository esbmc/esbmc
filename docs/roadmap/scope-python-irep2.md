# Scope — Phase 9, the Python frontend under `--python-irep2-adjust-only`

Phase 7 (clang-cpp) closed at 3095 of 3095 corpus rows agreeing
(`scope-clang-cpp-irep2.md` §8.5). This file records Phase 9 the same way: by
measurement, cause by cause.

## 1. The census, and the first cause it found

`scripts/irep2-migration/python_full_census.sh` is the C++ harness pointed at
`regression/python`, `python-contracts` and `python-coverage` under
`--python-irep2-adjust-only`. It is resumable, because at the rate this machine
manages — under one row per second, two ESBMC runs per row, `python3` forked
twice — 5101 rows do not finish in a sitting.

**Partial, 274 rows measured:** 268 agree, 4 skipped, **2 crash** — `class10` and
`class12`, both

```
ERROR: uncaught exception [bitwuzla]: terms with mismatching sort at indices 0 and 1
```

That is a *partial* number and is quoted as one. What it bought is the cause.

### 1.1 A string appended to a list, declared as an integer

The two rows share one declaration. Instrumented at the point the solver builds
the equality, the mismatching assignment is

```
lhs  main.py:78append_value409 : unsignedbv 64
rhs  constant_array { 102, 114, 111, 109, 95, 111, 98, 106, 52, 0 }   // "from_obj4"
```

and the GOTO programs differ in exactly that instruction:

```
legacy  ASSIGN main.py:78append_value409=(unsigned long int)(&{ 102, … , 0 }[0]);
flag    ASSIGN main.py:78append_value409={ 102, … , 0 };
```

A list element is stored as a pointer-sized integer, so appending a string
literal declares `unsigned long v = "…"`. Legacy decays the array to `&arr[0]`
and converts to the integer — `c_typecastt::do_typecast`'s array case followed by
the integer conversion. `python_adjust` had the decay for a *pointer* target
(the `char *` case) but not for an integer one, so the array reached the solver
whole and its sort disagreed with the declaration's.

Fixed by converting the initialiser to the declared type, which is
`clang_c_adjust_irep2::adjust_decl_init` narrowed to the one shape the Python
converter builds. Both rows now agree.

### 1.2 Three things this cost, worth recording

- **`--goto-functions-only` prints to stderr.** The first diff of the two GOTO
  programs came back empty because the redirect captured stdout only, which read
  as "the programs are identical" and sent the diagnosis into the symbol table for
  two rounds. The dumps differ in 414 lines.
- **Two plausible fixes were refuted before this one.** Keeping the literal's
  by-name struct type trips the pass's own exit invariant
  (`struct literal with by-name type 'tag-ZeroDivisionError'`), and skipping the
  literal's re-padding changes nothing. Both were measured and reverted.
- **The reduced reproducer did not bite.** A plain `values.append("…")`, a
  class-attribute list, and both with the incremental-SMT flags all pass with the
  arm gated off: the declaration shape needs more of `class10` than three
  attempts could isolate. The pinning pair is therefore a flag-carrying copy of
  that corpus row, `irep2_only_class_attr_append{,_fail}` — both halves crash with
  the arm gated off, which the CORE rows themselves cannot show, since they do not
  carry the flag.

## 2. Next

Resume the census. 274 of 5101 rows is 5%, and the C++ experience says the
residue is in families that only a full sweep names.
