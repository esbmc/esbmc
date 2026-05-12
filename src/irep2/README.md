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

## Reference: types

Defined by `ESBMC_LIST_OF_TYPES` in `irep2.h`; declared in `irep2_type.h`.

| Kind | Description |
|------|-------------|
| `bool` | Boolean type. No payload. |
| `empty` | "Void" — used for void pointers, void function returns, statements. |
| `symbol` | Placeholder/symbolic type used while linking, or for recursive references inside structs/arrays. |
| `struct` | C `struct` and C++ class data. Carries member types, names, pretty names, struct name, and `packed` flag. |
| `union` | C `union`. Same shape as `struct` (shares `struct_union_data`). |
| `code` | Function type: argument types, argument names, return type, ellipsis flag. |
| `array` | Fixed, infinite, or dynamically-sized array. Holds `subtype`, `array_size`, and `size_is_infinite`. |
| `vector` | SIMD-style fixed-size vector. Same backing data as `array` but distinct semantics. |
| `pointer` | Pointer to `subtype`; optional CHERI capability flag. |
| `unsignedbv` | Unsigned bitvector of given `width`. |
| `signedbv` | Signed bitvector of given `width`. |
| `fixedbv` | Fixed-point bitvector — total width split into integer and fraction bits. |
| `floatbv` | IEEE-754 floating-point — fraction-bit and exponent-bit counts. |
| `complex` | C `_Complex` — pair of identical scalar components (shares `struct_union_data`). |
| `cpp_name` | C++ qualified name with template arguments; used transiently by the C++ frontend. |

## Reference: expressions

Defined by `ESBMC_LIST_OF_EXPRS` in `irep2.h`; declared in `irep2_expr.h`. Every
expression has a `type` and an `expr_id`. Statement-shaped nodes (the `code_*`
family) carry `empty` type — they appear inside GOTO programs rather than
inside an expression tree.

### Constants

| Kind | Description |
|------|-------------|
| `constant_int` | Arbitrary-precision integer (`BigInt`), clipped to the bitvector type's width on simplification. |
| `constant_fixedbv` | Fixed-point literal (`fixedbvt`). |
| `constant_floatbv` | IEEE-754 floating-point literal (`ieee_floatt`). |
| `constant_bool` | Boolean literal (`true` / `false`). |
| `constant_string` | String literal (stored as an `irep_idt`). |
| `constant_struct` | Struct literal — one operand per member, in declaration order. |
| `constant_union` | Union literal — single active member and value. |
| `constant_array` | Array literal — one operand per element. |
| `constant_vector` | Vector literal — same shape as `constant_array`. |
| `constant_array_of` | Array filled uniformly with one value (cheap, scalable). |

### Symbols and casts

| Kind | Description |
|------|-------------|
| `symbol` | Reference to a named program variable (by `irep_idt`); type-checked against the symbol table. |
| `typecast` | Value-preserving conversion (e.g. int↔float, widening, narrowing). |
| `bitcast` | Reinterpret bits of one type as another type of equal width (no value conversion). |
| `nearbyint` | Round float to nearest integer per a rounding mode. |

### Control and equality

| Kind | Description |
|------|-------------|
| `if` | If-then-else expression: `cond ? a : b`. |
| `equality` | `a == b`. Boolean result. |
| `notequal` | `a != b`. Boolean result. |
| `lessthan` | `a < b`. Boolean result. |
| `greaterthan` | `a > b`. Boolean result. |
| `lessthanequal` | `a <= b`. Boolean result. |
| `greaterthanequal` | `a >= b`. Boolean result. |
| `cmp_three_way` | C++20 `a <=> b`; result is the appropriate comparison-category struct. |

### Logical (boolean)

| Kind | Description |
|------|-------------|
| `not` | Logical negation `!a`. |
| `and` | Logical and `a && b`. |
| `or` | Logical or `a \|\| b`. |
| `xor` | Logical xor. |
| `implies` | Logical implication `a -> b`. |

### Bitwise

| Kind | Description |
|------|-------------|
| `bitand` | Bitwise AND. |
| `bitor` | Bitwise OR. |
| `bitxor` | Bitwise XOR. |
| `bitnand` | Bitwise NAND. |
| `bitnor` | Bitwise NOR. |
| `bitnot` | Bitwise NOT. |
| `shl` | Logical shift left. |
| `lshr` | Logical shift right (zero-fill). |
| `ashr` | Arithmetic shift right (sign-fill). |

### Arithmetic (integer/fixed-point)

| Kind | Description |
|------|-------------|
| `neg` | Arithmetic negation. |
| `abs` | Absolute value. |
| `add` | Addition. |
| `sub` | Subtraction. |
| `mul` | Multiplication. |
| `div` | Division (semantics follow operand type). |
| `modulus` | Remainder / modulus. |

### IEEE-754 floating-point

Distinct from the integer ops because they carry a rounding mode and obey
IEEE semantics (NaN, infinities, signed zero).

| Kind | Description |
|------|-------------|
| `ieee_add` | IEEE addition with rounding mode. |
| `ieee_sub` | IEEE subtraction. |
| `ieee_mul` | IEEE multiplication. |
| `ieee_div` | IEEE division. |
| `ieee_fma` | IEEE fused multiply-add `(x*y)+z` rounded once. |
| `ieee_sqrt` | IEEE square root. |
| `isnan` | True iff operand is NaN. |
| `isinf` | True iff operand is ±infinity. |
| `isnormal` | True iff operand is a normal float. |
| `isfinite` | True iff operand is finite (not NaN or infinity). |
| `signbit` | Sign bit of the operand (int32 result). |

### Bit-level utilities

| Kind | Description |
|------|-------------|
| `popcount` | Population count (number of set bits). |
| `bswap` | Byte-swap (endianness reversal). |
| `concat` | Concatenate two unsigned bitvectors into a wider one. |
| `extract` | Slice bits `[upper:lower]` out of a bitvector. |

### Overflow checks

| Kind | Description |
|------|-------------|
| `overflow` | True iff the wrapped `add`/`sub`/`mul` overflows its operand width. |
| `overflow_cast` | True iff casting the operand to a narrower bitvector overflows. |
| `overflow_neg` | True iff negating the operand overflows (e.g. `INT_MIN`). |

### Pointers and the memory model

| Kind | Description |
|------|-------------|
| `address_of` | `&expr` — produces a pointer to its operand. |
| `dereference` | `*ptr` — expanded by symex into an if-then-else over candidate objects. |
| `same_object` | True iff two pointers refer to the same memory object. |
| `pointer_offset` | Byte offset component of a pointer. |
| `pointer_object` | Object-identifier component of a pointer. |
| `pointer_capability` | CHERI capability metadata of a pointer. |
| `valid_object` | True iff a pointer's target object is live (not freed, in scope). |
| `invalid_pointer` | True iff a pointer is structurally invalid. |
| `null_object` | Sentinel used by pointer analysis to represent the null target. |
| `dynamic_object` | Sentinel for a heap-allocated object (used by pointer analysis). |
| `dynamic_size` | Allocated size of a dynamic object (looked up from a model array at symex). |
| `deallocated_obj` | True iff a pointer's target has been freed (use-after-free probe). |
| `races_check` | Data-race check predicate used by the concurrent model checker. |
| `object_descriptor` | (Root object, offset, alignment) triple used during pointer reasoning. |
| `unknown` | Pointer-analysis sentinel: target is unknown. |
| `invalid` | Pointer-analysis sentinel: pointer is invalid. |

### Aggregate access and update

| Kind | Description |
|------|-------------|
| `index` | Element access `a[i]` on an array or vector. |
| `member` | Field access `s.field` on a struct/union. |
| `member_ref` | C++ reference-style member access. |
| `ptr_mem` | C++ pointer-to-member access `obj.*pm`. |
| `with` | Functional update — produce a new aggregate with one element/field replaced. |
| `byte_extract` | Extract a byte from a value at a runtime offset (used in byte-level memory model). |
| `byte_update` | Update a byte at a runtime offset, producing a new value. |

### Quantifiers

| Kind | Description |
|------|-------------|
| `forall` | Universal quantification `∀ sym. predicate`. |
| `exists` | Existential quantification `∃ sym. predicate`. |

### Python-specific predicates

Emitted by the Python frontend; lowered later in the pipeline.

| Kind | Description |
|------|-------------|
| `isinstance` | Python `isinstance(value, type)`. |
| `hasattr` | Python `hasattr(obj, attr)`. |
| `isnone` | Python `value is None` (and equivalents). |

### CHERI capabilities

| Kind | Description |
|------|-------------|
| `capability_base` | Base address of a CHERI capability. |
| `capability_top` | Top address (one past the last accessible byte) of a CHERI capability. |

### Side effects

| Kind | Description |
|------|-------------|
| `sideeffect` | Catch-all for effectful operations during frontend lowering: `malloc`/`alloca`, nondet allocation, embedded function calls. Carries an `allockind` discriminator. Most are flattened before GOTO. |

### GOTO statements (`code_*`)

These appear as instructions inside a GOTO program rather than inside an
ordinary expression tree. Their type is `empty` unless noted.

| Kind | Description |
|------|-------------|
| `code_block` | Sequence of statements. |
| `code_assign` | `target = source`. |
| `code_init` | Initialisation assignment (specialised `code_assign` — relevant for object lifetime). |
| `code_decl` | Declaration of a local with a given type and name (starts its lifetime). |
| `code_dead` | End of a local's lifetime — symex stops trusting reads through it. |
| `code_return` | `return expr;` (or `return;` if operand is nil). |
| `code_skip` | No-op. |
| `code_goto` | Unconditional jump to a label (`irep_idt`). Conditional gotos live on the GOTO instruction itself. |
| `code_function_call` | Function call `ret = func(args)`. |
| `code_free` | `free(ptr)` — releases a heap object. |
| `code_printf` | Captured `printf`/format call (preserves the format string). |
| `code_expression` | A statement consisting of a single expression evaluated for its side effects. |
| `code_comma` | C comma operator — evaluate left, then right; result is right. |
| `code_asm` | Inline assembly (the string is preserved; semantics are usually opaque). |

### C++ exception / delete statements

| Kind | Description |
|------|-------------|
| `code_cpp_delete` | C++ `delete expr;`. |
| `code_cpp_del_array` | C++ `delete[] expr;`. |
| `code_cpp_throw` | `throw expr;` with a list of types being thrown. |
| `code_cpp_catch` | `catch (...)` clause with a list of catchable types. |
| `code_cpp_throw_decl` | Function-level `throw(...)` declaration (start marker). |
| `code_cpp_throw_decl_end` | Matching end marker for `code_cpp_throw_decl`. |

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
