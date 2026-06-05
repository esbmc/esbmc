# IRep2 — ESBMC's Internal Representation

`irep2` is ESBMC's typed, intrusively reference-counted, copy-on-write
internal representation for **expressions** (`expr2t`) and **types**
(`type2t`). It is the data structure every frontend lowers to, every
transformation rewrites, and every backend (symex, SMT, goto2c) consumes.
It replaces the older "stringy" `irept` for the verification pipeline;
conversions live in `util/migrate.{h,cpp}`.

## Design at a glance

- **Typed class hierarchy.** Each kind of expression or type is its own
  C++ class (e.g. `add2t`, `symbol2t`, `signedbv_type2t`). A `type_ids` /
  `expr_ids` enum on the base class lets code dispatch without RTTI.
- **`expr2tc` / `type2tc` containers.** Hand-rolled smart pointers built
  on an intrusive atomic refcount sitting on `irep2t` itself: one
  allocation per node, no separate control block. They implement
  **copy-on-write** — the const accessors share the pointee, the
  non-const `get()` / `operator->` / `operator*` call `detach()` first,
  which clones into a fresh refcount-1 object when the count is > 1.
  This keeps containers cheap to copy and pass by value while preserving
  value semantics on mutation.
- **Construction.** Use `make_irep<T>(args...)` (or any of the generated
  `<name>2tc(args...)` factories that wrap it). The factory `new`s the
  node and hands it to a freshly-constructed container that adopts and
  increments the refcount.
- **Hash-consing friendly.** Each node caches its CRC in a
  `std::atomic<size_t>` (`irep2t::crc_val`); `0` means "not yet
  computed". Readers do an acquire load and skip the recompute on a hit;
  producers compute on a local and release-store. Mutation invalidates
  the cache via a relaxed store. `operator==` checks pointer identity
  first, then falls back to structural comparison.
- **Threading contract.** Single-writer / thread-confined. The refcount
  is atomic so containers may be dropped from any thread, but the
  pointee may have at most one mutator at a time. In debug builds the
  base class carries a writer-thread stamp that the container updates
  on every mutable access; a mismatch fires an assertion.
- **Generated boilerplate.** The full list of IR nodes lives in two
  per-family manifest files:
  - `expr_kinds.inc` — every expression kind (`add`, `if`, `symbol`,
    `code_assign`, ...).
  - `type_kinds.inc` — every type kind (`bool`, `signedbv`, `pointer`,
    `struct`, ...).

  Each entry is `IREP2_EXPR(kind, "pretty_name")` /
  `IREP2_TYPE(kind, "pretty_name")`. Consumers (the `expr_ids` /
  `type_ids` enums, the per-kind forward declarations, the
  `is_*` / `to_*` / `try_to_*` predicate generators, the pretty-name
  tables) `#include` the matching `.inc` with a redefining macro.

## File layout

| File | Contents |
|------|----------|
| `irep2.h` | Base classes `irep2t`, `type2t`, `expr2t`; the `irep_container` smart pointer (alias `expr2tc` / `type2tc`); the `make_irep` factory; `function_ref`; `hash_combine`; checked-cast helpers; the switch-on-id dispatchers for `cmp`/`lt`/`clone`/etc. |
| `expr_kinds.inc` / `type_kinds.inc` | Manifest of node kinds in declaration order. Single source of truth. |
| `irep2_type.h` / `irep2_type.cpp` | Concrete type classes (`bool_type2t`, `signedbv_type2t`, `array_type2t`, `pointer_type2t`, `struct_type2t`, ...). Each kind inherits directly from `type2t` and owns its fields. Free family helpers in the same header (`struct_union_members`, `array_or_vector_subtype`, ...) provide uniform field access when the caller doesn't care which specific kind it is. |
| `irep2_expr.h` / `irep2_expr.cpp` | Concrete expression classes (`constant_int2t`, `symbol2t`, `add2t`, `if2t`, `code_assign2t`, ...). Each kind inherits directly from `expr2t`. |
| `irep2_utils.h` | Inline predicates and helpers (`is_bv_type`, `is_number_type`, `is_scalar_type`, simplification helpers). |
| `irep2_dispatch.h` | Generic `generic_*<K>` helpers that walk a kind's `K::fields` tuple via `std::apply` to implement cmp/lt/crc/tostring/clone/get_sub_expr/foreach_operand uniformly, plus the per-field-type overloads they invoke (`do_type_cmp`, `do_type_lt`, `do_type_crc`, `type_to_string`, `do_get_sub_expr`, `call_*_delegate`). Switch dispatchers on `expr2t`/`type2t` route to these. |
| `irep2_utils.cpp` | Definitions for the predicates and dispatch-catalogue overloads declared in `irep2_utils.h` and `irep2_dispatch.h`. |
| `CMakeLists.txt` | Builds the `irep2` static library; depends on `bigint` and `fmt`. No Boost. |

## Anatomy of a node

Each concrete kind inherits directly from `expr2t` (or `type2t`, with a
small set of shared `*_data` intermediates on the type side) and
declares its fields plus two static class members:

```cpp
struct not2t : expr2t {
  expr2tc value;
  not2t(const type2tc &t, const expr2tc &v) : expr2t(t, not_id), value(v) {}
  static constexpr auto fields = std::make_tuple(&not2t::value);
  static std::string field_names[esbmct::num_type_fields];
};
```

The generic helpers in `irep2_dispatch.h` walk `K::fields` via
`std::apply` to implement `cmp`, `lt`, `clone`, `do_crc`, `hash`,
`tostring`, `get_sub_expr`, and `foreach_operand` once for every kind.
The switch-on-id dispatchers on `expr2t` / `type2t` (driven by the
`expr_kinds.inc` / `type_kinds.inc` X-macro manifests) route each call
to the matching `generic_*<K>`.

Net effect: adding a new node is a single-line manifest entry plus a
concrete class with `fields` / `field_names`. Comparison, hashing,
pretty-printing, and operand iteration are generated by the dispatchers
for free.

The `fields` tuple is canonical for the kind's primary constructor
argument order, after the type slot (so `K(type, f1, f2, ...)` matches
`K::fields = make_tuple(&K::f1, &K::f2, ...)` — or
`make_tuple(&expr2t::type, &K::f1, &K::f2, ...)` for kinds where the
type is itself part of the structural identity). `get_sub_expr(i)`
indexes into this tuple after skipping non-expr2tc fields, so the tuple
order is observable to callers.

## Working with containers

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

- `is_*2t` / `to_*2t` / `try_to_*2t` are generated from
  `expr_kinds.inc` / `type_kinds.inc`; never `dynamic_cast` directly.
  `to_*2t` throws `irep2_cast_error` (a `std::logic_error` subclass) on
  a mismatched kind; `try_to_*2t` returns `nullptr` instead.
- Treat `expr2tc` like a value: pass by const-ref where possible, copy
  where you need a private mutation point. The container will detach as
  needed.
- Calling `crc()` is safe to memoise on; mutation invalidates the cache.
- `simplify()` on a container delegates to the node's `do_simplify`. A
  nil return means "no further simplification."

## Adding a new node — checklist

1. Append a row to `expr_kinds.inc` (or `type_kinds.inc`):
   `IREP2_EXPR(my_kind, "my_kind")` / `IREP2_TYPE(my_kind, "my_kind")`.
   The pretty-name string is what `pretty()` and diagnostic messages
   print; it usually matches the identifier verbatim but may diverge
   (`null_object` prints as `NULL-object`).
2. Define the `<kind>2t` class in `irep2_expr.h` / `irep2_type.h`:
   inherit from `expr2t` / `type2t` (or a shared `*_data` base on the
   type side), declare each field, write a primary constructor
   `(type, field1, field2, ...)`, and add a `static constexpr auto
   fields = std::make_tuple(&kind2t::field1, ...)` tuple in the same
   order as the constructor parameters.
3. Add the `field_names` table for the kind in `irep2_expr.cpp` /
   `irep2_type.cpp` — one string per `fields` entry, in tuple order.
4. If the node has semantics worth modelling, teach `do_simplify`
   (defined out-of-line in `src/util/expr_simplifier.cpp` for
   non-trivial nodes) and the relevant lowering passes (symex, SMT
   conversion in `src/solvers/`, `goto2c`) about it.

## Related code

- `util/migrate.{h,cpp}` — bidirectional conversion between legacy
  string-based `irept`/`exprt` and `irep2tc`/`expr2tc`. Frontends emit
  legacy ireps that get migrated before symex.
- `src/goto-symex/`, `src/solvers/` — primary consumers; expect to walk
  `expr2tc` trees via `foreach_operand` and the `is_*` / `to_*` accessors.
- `src/util/std_expr.h`, `src/util/std_types.h` — legacy counterparts on the
  string-irep side.

## Grammar

The two manifests (`expr_kinds.inc` / `type_kinds.inc`) plus each kind's
`fields` tuple define an abstract grammar for the IR. The productions
below are that grammar; the *Reference* tables that follow enumerate every
terminal in prose. This is abstract syntax — the structure walked by
`get_sub_expr`, the comparison/crc dispatchers, and `clone` — not a
surface syntax any parser reads.

**Notation.** `::=` is a production, `|` alternation, `( … )` grouping,
`{ x }` zero-or-more, `[ x ]` optional, `"lit"` a terminal keyword (the
identifier as it appears in the `.inc` manifest). Lower-case words are
non-terminals; leaf terminals (`nat`, `name`, `int-lit`, …) are defined
at the end. `; …` is a comment. Every expression carries a result type,
written `«T»` after the keyword: `«type»` when the type is a free operand,
or a concrete type when the kind fixes it (`«bool»` for predicates,
`«empty»` for statements, `«pointer»` for `address_of`). Operand lists
appear in **`fields`-tuple order** — the order `get_sub_expr` and the
dispatchers walk.

### Types

```
type        ::= "bool"
              | "empty"
              | "symbol" "(" name ")"
              | aggregate-type
              | "code" "(" [ type { "," type } ] [ "," "..." ] ")" "->" type
              | "array" "(" type "," ( expr | "infinite" ) ")"
              | "vector" "(" type "," expr ")"
              | "pointer" "(" type [ "," "provenance" ] ")"
              | "unsignedbv" "(" nat ")"
              | "signedbv" "(" nat ")"
              | "fixedbv" "(" nat "," nat ")"     ; total width, integer bits
              | "floatbv" "(" nat "," nat ")"     ; fraction bits, exponent bits
              | "complex" "(" type ")"
              | "cpp_name" "(" name [ "<" type { "," type } ">" ] ")"

aggregate-type ::= ( "struct" | "union" ) name
                   "{" [ member { "," member } ] "}" [ "packed" ]
member      ::= name ":" type
```

`code` stores `argument_names` parallel to its argument types and an
`ellipsis` flag (the `...`). `struct`/`union` additionally carry
`member_pretty_names` parallel to `member` (elided above). An `array`
whose `size_is_infinite` flag is set has size `infinite` and a nil
`array_size`; a non-constant `array_size` is a dynamically-sized array.

### Expressions

```
expr        ::= constant | symbol | cast | control | relation | logical
              | bitwise | arithmetic | ieee | bitlevel | overflow
              | pointer-expr | aggregate-expr | quantifier | python-pred
              | cheri | sideeffect | statement
```

Constants:

```
constant    ::= "constant_int"      «type» int-lit
              | "constant_fixedbv"  «type» fixedbv-lit
              | "constant_floatbv"  «type» float-lit
              | "constant_bool"     «bool» bool-lit
              | "constant_string"   «type» string-lit
              | "constant_struct"   «type» "{" [ expr { "," expr } ] "}"
              | "constant_union"    «type» name "=" "{" [ expr { "," expr } ] "}"
              | "constant_array"    «type» "{" [ expr { "," expr } ] "}"
              | "constant_vector"   «type» "{" [ expr { "," expr } ] "}"
              | "constant_array_of" «type» "(" expr ")"
```

Symbols and casts:

```
symbol      ::= "symbol" «type» name
cast        ::= "typecast"  «type» "(" expr "," rmode ")"
              | "bitcast"   «type» "(" expr ")"
              | "nearbyint" «type» "(" expr "," rmode ")"
```

Control, comparison and logical:

```
control     ::= "if" «type» "(" expr "," expr "," expr ")"   ; cond, then, else
relation    ::= relop «bool» "(" expr "," expr ")"
              | "cmp_three_way" «type» "(" expr "," expr ")"
relop       ::= "equality" | "notequal" | "lessthan" | "greaterthan"
              | "lessthanequal" | "greaterthanequal"
logical     ::= "not" «bool» "(" expr ")"
              | logop2 «bool» "(" expr "," expr ")"
logop2      ::= "and" | "or" | "xor" | "implies"
```

Bitwise and arithmetic:

```
bitwise     ::= "bitnot" «type» "(" expr ")"
              | bitop2 «type» "(" expr "," expr ")"
bitop2      ::= "bitand" | "bitor" | "bitxor" | "shl" | "lshr" | "ashr"
arithmetic  ::= aunop «type» "(" expr ")"
              | abinop «type» "(" expr "," expr ")"
aunop       ::= "neg" | "abs"
abinop      ::= "add" | "sub" | "mul" | "div" | "modulus"
```

IEEE-754 floating point:

```
ieee        ::= ieee-bin   «type» "(" rmode "," expr "," expr ")"
              | "ieee_fma"  «type» "(" expr "," expr "," expr "," rmode ")"
              | "ieee_sqrt" «type» "(" expr "," rmode ")"
              | fp-class    «bool» "(" expr ")"
              | "signbit"   «type» "(" expr ")"
ieee-bin    ::= "ieee_add" | "ieee_sub" | "ieee_mul" | "ieee_div"
fp-class    ::= "isnan" | "isinf" | "isnormal" | "isfinite"
```

Bit-level utilities and overflow checks:

```
bitlevel    ::= "popcount" «type» "(" expr ")"
              | "bswap"    «type» "(" expr ")"
              | "concat"   «type» "(" expr "," expr ")"
              | "extract"  «type» "(" expr "," nat "," nat ")"  ; value, upper, lower
overflow    ::= "overflow"      «bool» "(" expr ")"  ; operand is an add/sub/mul
              | "overflow_cast" «bool» "(" expr "," nat ")"  ; operand, target width
              | "overflow_neg"  «bool» "(" expr ")"
```

Pointers and the memory model:

```
pointer-expr ::= "address_of"         «pointer» "(" expr ")"
               | "dereference"        «type» "(" expr ")"
               | "same_object"        «bool» "(" expr "," expr ")"
               | "pointer_offset"     «type» "(" expr ")"
               | "pointer_object"     «type» "(" expr ")"
               | "pointer_capability" «type» "(" expr ")"
               | "valid_object"       «bool» "(" expr ")"
               | "invalid_pointer"    «bool» "(" expr ")"
               | "deallocated_obj"    «bool» "(" expr ")"
               | "races_check"        «bool» "(" expr ")"
               | "dynamic_size"       «type» "(" expr ")"
               | "dynamic_object"     «type» "(" expr "," bool-lit "," bool-lit ")"
               | "object_descriptor"  «type» "(" expr "," expr "," nat ")"
               | "null_object"        «type»
               | "unknown"            «type»
               | "invalid"            «type»
```

Aggregate access and update:

```
aggregate-expr ::= "index"        «type» "(" expr "," expr ")"  ; array, index
                 | "member"       «type» "(" expr "," name ")"  ; aggregate, field
                 | "member_ref"   «type» "(" name ")"
                 | "ptr_mem"      «type» "(" expr "," expr ")"
                 | "with"         «type» "(" expr "," field "," expr ")"  ; src, field, val
                 | "byte_extract" «type» "(" expr "," expr "," endian ")"
                 | "byte_update"  «type» "(" expr "," expr "," expr "," endian ")"
field          ::= name | expr     ; field name for struct/union, index for array
endian         ::= "little" | "big"
```

Quantifiers, Python predicates and CHERI:

```
quantifier  ::= ( "forall" | "exists" ) «bool» "(" symbol "," expr ")"  ; bound var, body
python-pred ::= ( "isinstance" | "hasattr" | "isnone" ) «bool» "(" expr "," expr ")"
cheri       ::= ( "capability_base" | "capability_top" ) «type» "(" expr ")"
```

Side effects:

```
sideeffect  ::= "sideeffect" «type» "(" expr "," expr "," "[" { expr } "]"
                              "," type "," allockind ")"  ; operand, size, args, alloc-type
allockind   ::= "malloc" | "realloc" | "alloca" | "cpp_new" | "cpp_new_arr"
              | "nondet" | "va_arg" | "printf2" | "function_call"
              | "preincrement" | "postincrement" | "predecrement"
              | "postdecrement" | "old_snapshot" | "assigns_target"
```

GOTO statements (`code_*`):

```
statement   ::= "code_block"         «empty» "(" { statement } ")"
              | "code_assign"        «empty» "(" expr "," expr ")"  ; target, source
              | "code_decl"          «type»  name
              | "code_dead"          «type»  name
              | "code_return"        «empty» "(" expr ")"
              | "code_skip"          «type»
              | "code_free"          «empty» "(" expr ")"
              | "code_goto"          «empty» name                   ; target label
              | "code_function_call" «empty» "(" expr "," expr ","
                                              "[" { expr } "]" ")"  ; ret, func, args
              | "code_printf"        «empty» "(" "[" { expr } "]" ")"
              | "code_expression"    «empty» "(" expr ")"
              | "code_comma"         «type»  "(" expr "," expr ")"
              | "code_asm"           «type»  string-lit
              | cpp-statement
cpp-statement ::= "code_cpp_delete"         «empty» "(" expr ")"
                | "code_cpp_del_array"      «empty» "(" expr ")"
                | "code_cpp_throw"          «empty» "(" expr "," "[" { name } "]" ")"
                | "code_cpp_catch"          «empty» "(" "[" { name } "]" ")"
                | "code_cpp_throw_decl"     «empty» "(" "[" { name } "]" ")"
                | "code_cpp_throw_decl_end" «empty» "(" "[" { name } "]" ")"
```

Leaf terminals:

```
nat         ::= unsigned decimal integer (a bit-width or field index)
int-lit     ::= arbitrary-precision signed integer (BigInt)
fixedbv-lit ::= fixed-point literal (fixedbvt)
float-lit   ::= IEEE-754 literal (ieee_floatt)
bool-lit    ::= "true" | "false"
string-lit  ::= interned string literal (irep_idt)
name        ::= interned identifier (irep_idt) — variable, field, label or type name
rmode       ::= expr   ; rounding-mode operand, int32; defaults to __ESBMC_rounding_mode
```

Notes:

- **Type slot.** The `«…»` annotation is the node's `type` field. Whether
  it participates in structural equality depends on whether
  `&expr2t::type` is the first entry of the kind's `fields` tuple (see
  *Anatomy of a node*); a few kinds — `constant_bool`, `not` — omit it.
- **Rounding mode.** Operand order follows `fields`, and the rounding-mode
  operand's position is not uniform: the binary `ieee_*` ops list it
  first, while `ieee_fma` / `ieee_sqrt` list it last.
- **No conditional `code_goto`.** `code_*` statements carry `empty` type
  and live inside GOTO programs. Conditional branches are encoded on the
  GOTO *instruction*, not as an expression, so only the unconditional
  `code_goto` appears here.

## Examples

Each example shows the source, the C++ construction with the generated
`*2tc` factories, and the resulting tree in the grammar notation above.
Assume `type2tc i32 = signedbv_type2tc(32);` throughout.

**Leaves — a variable and a literal.**

```cpp
expr2tc x   = symbol2tc(i32, "x");
expr2tc one = constant_int2tc(i32, BigInt(1));
```
```
symbol       «signedbv(32)» "x"
constant_int «signedbv(32)» 1
```

**Arithmetic — `x + 2`.** `add` takes its result type plus two operands:

```cpp
expr2tc two = constant_int2tc(i32, BigInt(2));
expr2tc sum = add2tc(i32, x, two);
```
```
add «signedbv(32)» (
  symbol       «signedbv(32)» "x",
  constant_int «signedbv(32)» 2)
```

**Comparison + ternary — `y > 2 ? y : 0`.** Relations have boolean
result and take only their two operands (no explicit type):

```cpp
expr2tc y    = symbol2tc(i32, "y");
expr2tc gt   = greaterthan2tc(y, two);
expr2tc zero = constant_int2tc(i32, BigInt(0));
expr2tc tern = if2tc(i32, gt, y, zero);
```
```
if «signedbv(32)» (
  greaterthan «bool» (symbol «signedbv(32)» "y",
                      constant_int «signedbv(32)» 2),
  symbol       «signedbv(32)» "y",
  constant_int «signedbv(32)» 0)
```

**Struct member — `p.y`** for `struct point { int x; int y; }`:

```cpp
type2tc point = struct_type2tc(
  {i32, i32},      // members
  {"x", "y"},      // member_names
  {"x", "y"},      // member_pretty_names
  "point",         // tag
  false);          // not packed
expr2tc p   = symbol2tc(point, "p");
expr2tc p_y = member2tc(i32, p, "y");   // result type is the member's type
```
```
member «signedbv(32)» (
  symbol «struct point {x: signedbv(32), y: signedbv(32)}» "p",
  "y")
```

**Array and pointers — `a[i]`, `&x`, `*p`.** Note `address_of2tc` takes
the *pointee* type as its first argument; the node's own type is the
pointer to it:

```cpp
expr2tc a    = symbol2tc(
  array_type2tc(i32, constant_int2tc(i32, BigInt(3)), false), "a");
expr2tc i    = symbol2tc(signedbv_type2tc(64), "i");
expr2tc elem = index2tc(i32, a, i);          // a[i]
expr2tc addr = address_of2tc(i32, x);        // &x : pointer(signedbv(32))
expr2tc p    = symbol2tc(pointer_type2tc(i32), "p");
expr2tc star = dereference2tc(i32, p);       // *p
```
```
index       «signedbv(32)»          (symbol «array(signedbv(32), 3)» "a",
                                     symbol «signedbv(64)» "i")
address_of  «pointer(signedbv(32))» (symbol «signedbv(32)» "x")
dereference «signedbv(32)»          (symbol «pointer(signedbv(32))» "p")
```

**Statements — a function body.** `code_*` nodes nest inside a
`code_block`; their type is `empty`:

```cpp
expr2tc body = code_block2tc(std::vector<expr2tc>{
  code_decl2tc(i32, "x"),
  code_assign2tc(x, one),
  code_decl2tc(i32, "y"),
  code_assign2tc(y, add2tc(i32, x, two)),
  code_return2tc(y),
});
```
```
code_block «empty» (
  code_decl   «signedbv(32)» "x",
  code_assign «empty» (symbol «signedbv(32)» "x", constant_int «signedbv(32)» 1),
  code_decl   «signedbv(32)» "y",
  code_assign «empty» (symbol «signedbv(32)» "y",
                       add «signedbv(32)» (symbol «signedbv(32)» "x",
                                           constant_int «signedbv(32)» 2)),
  code_return «empty» (symbol «signedbv(32)» "y"))
```

**Seeing the IR for real code.** `esbmc file.c --goto-functions-only`
dumps the GOTO program, where the same `code_*` / expression nodes print
in a compact concrete syntax. For

```c
int f(int *a, int i) {
  struct point p;
  int x = 1;
  int y = x + 2;
  p.y = y;
  int z = a[i];
  if (y > 2) return p.y;
  return z;
}
```

the listing (comments and `p`'s nondet-init trimmed) is:

```
DECL signed int x;
ASSIGN x=1;
DECL signed int y;
ASSIGN y=x + 2;
ASSIGN p.y=y;
DECL signed int z;
ASSIGN z=a[(signed long int)i];
IF !(y > 2) THEN GOTO 1
RETURN: p.y
1: RETURN: z
```

Reading it back to the grammar: `ASSIGN y=x + 2;` is
`code_assign(symbol "y", add(symbol "x", constant_int 2))`; `p.y` is
`member(symbol "p", "y")`; `a[(signed long int)i]` is
`index(symbol "a", typecast(symbol "i"))`; and the `IF … GOTO` is the
conditional branch carried on the GOTO instruction itself — not a
`code_goto` expression (see the grammar note above).

## Reference: types

Defined by `ESBMC_LIST_OF_TYPES` in `irep2.h`; declared in `irep2_type.h`.

| Kind | Description |
|------|-------------|
| `bool` | Boolean type. No payload. |
| `empty` | "Void" — used for void pointers, void function returns, statements. |
| `symbol` | Placeholder/symbolic type used while linking, or for recursive references inside structs/arrays. |
| `struct` | C `struct` and C++ class data. Carries member types, names, pretty names, struct name, and `packed` flag. |
| `union` | C `union`. Same field shape as `struct` (members, names, packed, ...). |
| `code` | Function type: argument types, argument names, return type, ellipsis flag. |
| `array` | Fixed, infinite, or dynamically-sized array. Holds `subtype`, `array_size`, and `size_is_infinite`. |
| `vector` | SIMD-style fixed-size vector. Same backing data as `array` but distinct semantics. |
| `pointer` | Pointer to `subtype`; optional CHERI capability flag. |
| `unsignedbv` | Unsigned bitvector of given `width`. |
| `signedbv` | Signed bitvector of given `width`. |
| `fixedbv` | Fixed-point bitvector — total width split into integer and fraction bits. |
| `floatbv` | IEEE-754 floating-point — fraction-bit and exponent-bit counts. |
| `complex` | C `_Complex` — pair of identical scalar components. Currently stored with the same `members`/`member_names`/... shape as `struct`/`union` so the SMT tuple lowering can treat it uniformly; a follow-up will redesign it as a primitive `subtype` field. |
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
- **CRC cache.** `crc_val` is a `std::atomic<size_t>` with `0` meaning
  "not yet computed". Readers do an acquire load and skip recompute on
  a hit; producers compute on a local and release-store the result.
  Concurrent readers should call `crc()`, not the raw field.
- **`type2t::get_width()` can throw.** Symbolic, infinite-sized, or
  dynamically-sized types raise `symbolic_type_excp` /
  `array_type2t::inf_sized_array_excp` / `dyn_sized_array_excp`. Callers
  that consume widths from arbitrary types must handle these.
- **Don't add IR kinds outside the preprocessor lists.** Vast amounts of
  generated code assume the master lists are the single source of truth.
