# func_internal_type_1 KNOWNBUG Investigation

## Current status (2026-04-13)

**Frontend crash FIXED.** The Solidity frontend now accepts `FunctionTypeName`
AST nodes for parameters and struct fields — they are lowered to an opaque
`void *` in `solidity_convert_type.cpp` (Pointer case), and indirect calls
through them are rewritten at the call site to a `nondet()` expression of the
declared return type (`solidity_convert_expr.cpp`).

The test stays **KNOWNBUG** because the Pyramid example asserts specific
computed values:

```solidity
assert(pyramid(4) == 14); // range -> map(square) -> reduce(sum)
```

With indirect calls returning nondet, `map(f)` and `reduce(f)` cannot compute
real values — the assertion fails as `VERIFICATION FAILED` instead of the
expected `VERIFICATION SUCCESSFUL`.

## What would fix the precision gap

Full support requires function-pointer lowering + call resolution in the
frontend. Approaches, roughly in order of cost:

1. **Source-level monomorphization (Hack 2 in the plan)**: when the converter
   sees `arr.map(square).reduce(sum)`, clone `map`/`reduce` into specialized
   copies `map__square`/`reduce__sum` with the callback inlined, and call the
   clones directly. Works whenever the callback is statically known at each
   call site (true for all Solidity docs examples). Estimated 200-400 lines
   plus call-site rewriting for method chains.

2. **Full function-pointer IR**: represent internal function types as
   `code_typet` pointers, emit an enum-dispatch table of candidate targets,
   and translate each indirect call to a switch over the candidates. More
   general but heavier.

## Related

- See `CLAUDE_Solidity.md` → "Function types (internal)" row.
- Same Hack-1 strategy will **not** unblock `func_external_type_1`: passing a
  function reference as an argument (`ORACLE_CONST.query("USD", this.callback)`)
  still crashes during argument marshalling in `get_non_library_function_call`
  — a separate code path that does not route through the Pointer type case.

## Date

2026-04-13 (Hack 1 implemented; frontend no longer crashes)
2026-04-12 (original KNOWNBUG report — frontend crash)
