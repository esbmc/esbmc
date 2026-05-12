# IRep2 — ESBMC's Internal Representation

`irep2` is ESBMC's typed, reference-counted, copy-on-write internal
representation for **expressions** (`expr2t`) and **types** (`type2t`). It is
the data structure every frontend lowers to, every transformation rewrites,
and every backend (symex, SMT, goto2c) consumes. It replaces the older
"stringy" `irept` for the verification pipeline; conversions live in
`util/migrate.{h,cpp}`.

## Design at a glance

- **Typed class hierarchy.** Each kind of expression or type is its own C++
  class (e.g. `add2t`, `symbol2t`, `signedbv_type2t`). A `type_ids` /
  `expr_ids` enum on the base class lets code dispatch without RTTI.
- **`expr2tc` / `type2tc` containers.** Thin wrappers around
  `std::shared_ptr` providing **copy-on-write**: the const accessors share
  the pointee, and the non-const `get()`/`operator->` automatically `detach()`
  into a fresh copy when the refcount is > 1. This keeps `expr2tc` cheap to
  copy and pass by value while preserving value semantics on mutation.
- **Hash-consing friendly.** Each node caches its CRC (`crc_val`, guarded by
  `crc_mutex`); the cache is invalidated on detach. `operator==` first checks
  pointer identity, then falls back to structural comparison.
- **Generated boilerplate.** The full list of IR nodes lives in two Boost
  preprocessor lists at the top of `irep2.h`:
  - `ESBMC_LIST_OF_EXPRS` — every expression kind (`add`, `if`, `symbol`,
    `code_assign`, ...).
  - `ESBMC_LIST_OF_TYPES` — every type kind (`bool`, `signedbv`, `pointer`,
    `struct`, ...).
  These lists drive forward declarations, the `expr_ids` / `type_ids` enums,
  `is_foo2t()` / `to_foo2t()` accessors, and template instantiations.

## File layout

| File | Contents |
|------|----------|
| `irep2.h` | Base classes `irep2t`, `type2t`, `expr2t`; the `irep_container` smart pointer (alias `expr2tc` / `type2tc`); master node-kind lists. |
| `irep2_type.h` / `irep2_type.cpp` | Concrete type classes (`bool_type2t`, `signedbv_type2t`, `array_type2t`, `pointer_type2t`, `struct_type2t`, ...). |
| `irep2_expr.h` / `irep2_expr.cpp` | Concrete expression classes (`constant_int2t`, `symbol2t`, `add2t`, `if2t`, `code_assign2t`, ...) and their *data* base classes (`constant_int_data`, `arith_2ops`, ...). |
| `irep2_utils.h` | Inline predicates and helpers (`is_bv_type`, `is_number_type`, `is_scalar_type`, `is_multi_dimensional_array`, simplification helpers). |
| `irep2_templates.h` | CRTP-style scaffolding (`register_irep_methods`, `do_type2string`, `do_get_sub_expr`) that materialises `pretty`, `cmp`, `lt`, `do_crc`, `hash`, `clone`, and operand iteration for each node. |
| `irep2_templates_expr.h`, `irep2_templates_types.h`, `irep2_template_utils.h`, `irep2_meta_templates.h` | Per-field trait machinery (`field_traits`, `expr2t_traits`, `type2t_traits`) and the metaprogramming that walks them. |
| `templates/*.cpp` | Out-of-line explicit instantiations of the per-node template methods (split into several TUs to keep compile times manageable). |
| `CMakeLists.txt` | Builds the `irep2` static library (depends on `bigint`, Boost, `fmt`, and privately `crypto_hash`). |

## Anatomy of a node

A concrete node is composed by inheritance from three layers:

1. A **data class** holds the fields and exposes them as `field_traits` so the
   metaprogramming can enumerate them — e.g. `constant_int_data` holds a
   `BigInt value` field.
2. An **`esbmct::expr2t_traits<...>`** typedef lists those fields. The
   templates in `irep2_templates*.h` consume this trait list to generate
   `clone`, `cmp`, `lt`, `do_crc`, `hash`, `tostring`, `foreach_operand`,
   `get_sub_expr`, etc.
3. The user-facing class (e.g. `constant_int2t`) inherits from the data class
   and registers itself via `irep_methods2`, picking up all the generated
   members.

Net effect: adding a new node is mostly a matter of (a) appending its name to
the master list in `irep2.h`, (b) declaring a data class with `field_traits`,
and (c) declaring the user-facing class. Comparison, hashing, pretty-printing
and operand iteration are generated for you.

## Working with `irep2tc` containers

```cpp
expr2tc lhs = symbol2tc(int_type, "x");
expr2tc rhs = constant_int2tc(int_type, BigInt(1));
expr2tc sum = add2tc(int_type, lhs, rhs);       // builds an add2t

if (is_add2t(sum))                               // generated predicate
{
  const add2t &a = to_add2t(sum);                // generated downcast
  expr2tc lhs_copy = a.side_1;                   // cheap: shares the pointee
}

// Mutation via non-const access transparently detaches:
to_add2t(sum).side_1 = constant_int2tc(int_type, BigInt(2));
```

Notes:

- `is_*2t` / `to_*2t` are generated from `ESBMC_LIST_OF_EXPRS` /
  `ESBMC_LIST_OF_TYPES`; never `dynamic_cast` directly.
- Treat `expr2tc` like a value: pass by const-ref where possible, copy where
  you need a private mutation point. The container will detach as needed.
- Calling `crc()` is safe to memoise on; mutation invalidates the cache.
- `simplify()` on a container delegates to the node's `do_simplify`. A nil
  return means "no further simplification."

## Adding a new node — checklist

1. Append the node name to `ESBMC_LIST_OF_EXPRS` (or `ESBMC_LIST_OF_TYPES`)
   in `irep2.h`.
2. Define a *data* class in `irep2_expr.h` / `irep2_type.h` listing each
   field with `esbmct::field_traits<...>` and aggregating them into a
   `traits` typedef.
3. Define the user-facing `foo2t` class (typically empty — it just inherits
   the data class and registers methods).
4. Add explicit template instantiations to the matching file under
   `templates/` so the generated methods get emitted out-of-line.
5. If the node has semantics worth modelling, teach `do_simplify` and the
   relevant lowering passes (symex, SMT conversion in `src/solvers/`,
   `goto2c`) about it.

## Related code

- `util/migrate.{h,cpp}` — bidirectional conversion between legacy
  string-based `irept`/`exprt` and `irep2tc`/`expr2tc`. Frontends emit
  legacy ireps that get migrated before symex.
- `src/goto-symex/`, `src/solvers/` — primary consumers; expect to walk
  `expr2tc` trees via `foreach_operand` and the `is_*` / `to_*` accessors.
- `src/util/std_expr.h`, `src/util/std_types.h` — legacy counterparts on the
  string-irep side.

## Gotchas

- **Detach semantics.** Holding a raw `T *` from `irep2tc::get()` (non-const)
  detaches; calling `get()` const does not. Don't cache the raw pointer
  across non-const accesses on shared containers.
- **CRC cache.** Any non-const access resets `crc_val` to 0. Concurrent
  readers of the same node must go through `crc()` (which locks
  `crc_mutex`), not the raw field.
- **`type2t::get_width()` can throw.** Symbolic, infinite-sized, or
  dynamically-sized types raise `symbolic_type_excp` /
  `array_type2t::inf_sized_array_excp` / `dyn_sized_array_excp`. Callers
  that consume widths from arbitrary types must handle these.
- **Don't add IR kinds outside the preprocessor lists.** Vast amounts of
  generated code assume the master lists are the single source of truth.
