# Pointer Analysis

Developer notes on ESBMC's memory model and the code under
`src/pointer-analysis/`. The directory provides two related services:

1. **Value-set tracking** — for every pointer-typed l1 variable, the set of
   data objects it may point at, with offset and alignment information
   (`value_set.{h,cpp}`, `value_set_domain.{h,cpp}`, `value_set_analysis.{h,cpp}`,
   `value_sets.h`).
2. **Pointer dereferencing** — given a points-to set and a dereference site,
   produce an expression whose value, under the SMT model, equals the data the
   pointer reads (or, for stores, writes through) at the requested offset and
   type (`dereference.{h,cpp}`, `goto_program_dereference.{h,cpp}`).

The same `value_sett` data structure is reused by goto-symex (live points-to
tracking during symbolic execution) and by `value_set_analysist` (a flow- and
context-insensitive fixpoint computed before symex via the
`static_analysist<value_set_domaint>` framework).

## Why this is non-trivial

The C standard requires every data object to have a byte representation:
arbitrary pointer arithmetic on a `char *` is well-defined, and `memcpy` works
by copying bytes. A direct encoding of this in SMT — a single byte array
covering all storage — is sound but tends to hurt solver performance.

ESBMC instead stores each l1 data object as a single SMT primitive of the
object's natural type (bitvector, float, bool, array, structure, or union).
The cost of this choice is that the dereferencing code must translate every
read or write through a pointer into an access on the primitive backing the
pointed-to object — including the cases where the access type, the pointer
type, and the primitive's type all disagree.

For example:

```c
int foo, bar;
char *baz = (nondet_bool()) ? (char *)&foo : (char *)&bar;
unsigned int idx = nondet_uint();
__ESBMC_assume(idx < sizeof(int));
baz += idx;
*baz = 1;
```

A store through `baz` must be lowered into an access into either `foo` or
`bar`'s SMT primitive, at a non-constant byte offset, writing a `char` into
an `int`. Three types are in play:

* the type of the right-hand side,
* the type of the pointer being dereferenced,
* the type of the SMT primitive backing the pointed-to data object.

## Value sets (`value_set.{h,cpp}`)

`value_sett` maps `(l1 identifier, suffix) → entryt`. An `entryt` holds an
`object_mapt` (an `unordered_map<unsigned, objectt>`) whose keys are indices
into the static `value_sett::object_numbering` and whose values are `objectt`
records carrying:

* `offset` (`BigInt`) and `offset_is_set` — fixed offset into the object, or
  flag indicating the offset is symbolic;
* `offset_alignment` (bytes) — lower bound on the alignment of the symbolic
  offset; never zero when `offset_is_set` is false.

The suffix uniquely identifies a sub-pointer within an l1 variable (e.g.
`.field_name` for a struct member, `[]` for an array of pointers — array
elements are not tracked individually).

`object_numbering` and `obj_numbering_refset` are `thread_local` so parallel
symex threads (e.g. `--k-induction-parallel`) don't race on the shared
numbering.

Key methods:

* `assign(lhs, rhs, add_to_sets)` — interprets an assignment and updates the
  pointing data. `add_to_sets=true` is the join semantics used by the static
  analysis.
* `get_value_set(expr, dest)` — returns, for an expression of pointer type,
  the set of object references it may evaluate to. Pointer arithmetic on the
  expression updates offsets; non-deterministic pointer arithmetic
  conservatively widens to "unknown offset" with a tracked alignment.
* `get_reference_set(expr, dest)` — returns the *referents* used in `expr`
  rather than the values reached through dereferences (e.g. for `&a->foo`).
* `make_union(other, keepnew)` — merge two `valuest` maps; used at CFG joins
  and during fixpoint computation.

Entries with identifier prefix `value_set::dynamic_object` represent
heap-allocated storage created during symbolic execution; the symbolic
heap-tracking logic ("black magic" in earlier comments) lives in
`do_free` and the dynamic-object handling inside `assign_rec`.

### Static analysis wrapper

`value_set_analysist` (a `static_analysist<value_set_domaint>`) computes a
fixed-point of the points-to map across the goto program. `value_set_domaint`
owns a `value_sett *` and forwards `merge` and `transform` to it. The
`value_setst` abstract base in `value_sets.h` is the lookup interface used by
clients that don't care whether the analysis is the static one or the
symbolic-execution one:

```cpp
class value_setst {
public:
  virtual void get_values(goto_programt::const_targett l,
                          const expr2tc &expr,
                          valuest &dest) = 0;
};
```

where `valuest` is `std::list<expr2tc>` containing
`object_descriptor2t`/`unknown2t`/`invalid2t` expressions.

## Dereferencing (`dereference.{h,cpp}`)

`dereferencet` is stateless w.r.t. the program — it holds only a namespace, a
context (for minting new symbols), command-line options, and a back-pointer
to a `dereference_callbackt`. The callback is the interface to the
environment hosting the dereference:

* `get_value_set(expr, value_set)` — points-to query;
* `dereference_failure(property, msg, guard)` — record a property violation
  (e.g. null deref, out-of-bounds, alignment);
* `dereference_assume(guard)` — record a derived assumption;
* `is_live_variable(sym)` — answer whether an l1 stack variable is still
  on the call stack (used by symex; the goto-program wrapper always returns
  true);
* `dump_internal_state(items)` — receive raw object/offset/guard triples in
  `INTERNAL` mode without building a reference;
* `rename(expr)` — optional rename hook (currently a no-op).

`goto_program_dereferencet` (in `goto_program_dereference.{h,cpp}`) is the
implementation used in front-of-symex passes such as `pointer_checks`. The
symex side has its own `dereference_callbackt` implementation; both share the
same `dereferencet` code.

### Modes

`dereferencet::modet` is a small bitfield:

* `op` ∈ {`READ`, `WRITE`, `FREE`, `INTERNAL`} — kind of access. `INTERNAL`
  asks the dereference code to populate `internal_items` and return nothing
  to the caller; only the callback receives data via `dump_internal_state`.
* `unaligned` — set when the access is known to be (potentially) unaligned,
  e.g. a member of a `__attribute__((packed))` struct, so the alignment
  checker should not fire.

Predicates (`is_read`, `is_write`, `is_free`, `is_internal`) compare on the
`op` field only — there is intentionally no `operator==`.

### Pipeline

The four phases listed in `dereference.h`:

#### 1. Surface expression walk — `dereference_expr`

Recurses over the expression looking for dereferences. Special cases:

* `and_id`, `or_id`, `if_id` → `dereference_guard_expr`. Each operand is
  dereferenced under the appropriate short-circuit guard so that assertions
  contributed by short-circuited operands don't fire spuriously.
* `address_of_id` → `dereference_addrof_expr`. Folds `&*p` to `p`, and
  rewrites `&base->m[i].n` into pointer arithmetic on a `uint8_t *`, avoiding
  an actual dereference where possible.
* `dereference_id` → recurse into the operand under mode `READ`, then call
  `dereference(value, type, guard, mode, expr2tc())`.
* `index_id` and `member_id` → `dereference_expr_nonscalar`. The result type
  at this level is a scalar — the chain of `member` and `index` operations
  applied on top of a dereference is collapsed and the eventual `dereference`
  call gets the cumulative bit-offset as `extra_offset`. This avoids
  materialising intermediate struct/array values, which would otherwise have
  to be reconstructed only to project a field out again.
* Otherwise: recurse into each operand.

`dereference_expr_nonscalar` walks the `member`/`index`/`typecast`/
`constant_union` chain rooted at `base`, accumulates the bit-offset via
`compute_pointer_offset_bits(base, &ns)`, and on reaching the
`dereference2t` calls `dereference(...)` with that bit-offset. If the
expression accessed traverses a packed struct member without alignment, the
mode's `unaligned` flag is set for the remainder of the walk (unless
`--no-align-check` is set).

#### 2. Per-object dispatch — `dereference`

```cpp
expr2tc dereference(const expr2tc &dest, const type2tc &type,
                    const guard2tc &guard, modet mode,
                    const expr2tc &extra_offset);
```

* `dest` — the (already-recursively-dereferenced) pointer expression.
* `type` — the desired result type. Followed via `ns.follow`.
* `guard` — execution-path guard for the dereference site.
* `mode` — `READ` / `WRITE` / `FREE` / `INTERNAL` (+ `unaligned`).
* `extra_offset` — additional offset in **bits** introduced by the lexical
  index/member chain (see phase 1); `nil` if none.

Steps:

1. Cast `dest` to a pointer type if it isn't one (defensive: nested
   dereferences may produce non-pointer expressions, which the rest of the
   pipeline still handles by failing into a free symbol).
2. `dereference_callback.get_value_set(src, points_to_set)` — fetch the set
   of candidate object references. Each is an `object_descriptor2t`,
   `unknown2t`, or `invalid2t`.
3. If any candidate is `unknown`/`invalid`, the set is treated as
   non-exhaustive and a *failed symbol* is seeded as the initial value
   (`make_failed_symbol`). Otherwise the chain starts empty so that the last
   `if-then-else` arm is unconditional.
4. For each candidate, call `build_reference_to(...)`. Successful results are
   chained into an `if(pointer_guard, this_value, accumulator)` cascade.
5. In `INTERNAL` mode, no chain is built; instead the per-object
   `(object, offset, guard)` triples accumulated in `internal_items` are
   handed to `dump_internal_state` and cleared.

Failed symbols are fresh free variables named `symex::invalid_object<N>`
(`invalid_counter` is `thread_local` to keep names unique under parallel
symex). They serve as well-typed placeholder rvalues so the formula remains
valid when the access is guaranteed to fail.

#### 3. Reference building — `build_reference_to` / `build_reference_rec`

`build_reference_to` is the per-candidate entry point. It:

* runs `check_pointer_alignment` for applicable modes;
* short-circuits `unknown`/`invalid` candidates into a `dereference_failure`
  via `deref_invalid_ptr`;
* short-circuits the `NULL` object: emits a null-deref failure for read/write
  modes, and a silent no-op for `FREE`/`INTERNAL` (freeing `NULL` is legal in
  C);
* computes the final offset in bits:
  - starts from `object_descriptor2t::offset` if constant;
  - otherwise uses `pointer_offset2tc(deref_expr)` and tracks the alignment
    advertised by the value-set entry (`alignment` falls back to 1 if the
    dereferenced expression is not a plain symbol);
  - converts bytes → bits and adds `lexical_offset`;
* builds `same_object2tc(deref_expr, &object)` as the per-candidate
  `pointer_guard` (also returned via out-parameter for the caller's
  if-chain);
* runs `valid_check` against the candidate object, then `bounds_check` /
  `check_data_obj_access` / `check_code_access` as applicable;
* finally calls `build_reference_rec(value, final_offset_in_bits, type,
  guard, mode, alignment_in_bits)`.

Provenance: when the dereferenced pointer's type has `carry_provenance`
(CHERI / CHERI-C), `bounds_check` is given the pointer expression so it can
emit provenance-aware bounds.

`build_reference_rec` is the dispatching switch. It encodes the cross-product
of (source kind, destination kind, offset kind) into a single integer key
from `target_flags`:

```text
flag_src_scalar = 0x01   flag_dst_scalar = 0x10   flag_is_const_offs = 0x80
flag_src_array  = 0x02   flag_dst_struct = 0x20   flag_is_dyn_offs   = 0x100
flag_src_struct = 0x04   flag_dst_union  = 0x40
flag_src_union  = 0x08
```

Each bit is distinct (no flag is zero) so that case labels never collide
when OR'd. Source category is read from `value->type`; destination from the
requested `type`; offset kind from whether `offset` is a `constant_int2t`.

Code-typed values and code-typed destinations are returned untouched. A
zero-bit destination type returns a `gen_zero(type)` (or a failed symbol for
a memberless union) — this is the fix for issue #723; without it, the
recursive paths reach the scalar base case and fail with a spurious width
mismatch.

The full dispatch table is the comment block immediately above
`build_reference_rec` in `dereference.cpp`. In summary:

| src \ dst | scalar (const off)              | scalar (dyn off)        | struct (const off)              | struct (dyn off)              | union (const off)              | union (dyn off)              |
|-----------|---------------------------------|-------------------------|---------------------------------|-------------------------------|--------------------------------|------------------------------|
| scalar    | `construct_from_const_offset`   | `construct_from_dyn_offset` | bitcast if off=0 & widths match | error                         | bitcast if off=0 & widths match | error                        |
| struct    | `construct_from_const_struct_offset` | `construct_from_dyn_struct_offset` | `construct_struct_ref_from_const_offset` | `construct_struct_ref_from_dyn_offset` | `construct_struct_ref_from_const_offset` | `construct_struct_ref_from_dyn_offset` |
| array     | `construct_from_array`          | `construct_from_array`  | `construct_struct_ref_from_const_offset_array` | `construct_struct_ref_from_dyn_offset` | `construct_struct_ref_from_const_offset_array` | `construct_struct_ref_from_dyn_offset` |
| union     | ad-hoc: pick the widest member, recurse | ad-hoc: pick the widest member, recurse | ad-hoc: pick the widest member, recurse | ad-hoc: pick the widest member, recurse | recurse with widest member | recurse with widest member |

`construct_struct_ref_*` returning a struct only happens when unavoidable
(e.g. structure assignment through a `char *`-backed allocation), and is
guarded by aggressive assertions — the preferred path is to dereference
directly into the requested scalar field via the lexical chain.

When the access cannot be expressed as a clean projection of the underlying
primitive — e.g. a misaligned read, or reading an `int` from an array of
`short` — the code falls back to **byte stitching**:
`extract_bytes` produces one expression per byte of the source primitive,
`stitch_together_from_byte_array` concatenates them in target endianness
(`is_big_endian`, derived from `config.ansi_c.endianess`), and
`extract_bits_from_byte_array` projects the bit-slice asked for. The
two overloads of `stitch_together_from_byte_array` cover, respectively, the
fixed-byte-count case and the array-source-with-(possibly-dynamic)-bit-offset
case.

#### 4. Assertions

Recorded via the callback as the dereferencing code proceeds:

* `valid_check` — object liveness (for stack variables, `is_live_variable`),
  freed-object access, write-to-string-literal.
* `bounds_check` — offset within object size (provenance-aware for CHERI).
* `check_data_obj_access` and `check_code_access` — type/representation
  constraints on the access.
* `check_alignment` / `check_pointer_alignment` — alignment of the offset
  against the access type, controlled by the `unaligned` mode flag.

All of these route through `dereference_failure(...)` →
`dereference_callback.dereference_failure(...)`. `--no-pointer-check` and
`--no-align-check` suppress whole assertion classes (queried inline in
`dereference.cpp`; `goto_program_dereferencet::dereference_failure` also
gates on `no-simplify`). Setting `block_assertions` on `dereferencet`
silently drops all dereference-failure assertions and is used for the
`INTERNAL` mode and similar query paths.

## Byte-layout invariant

The dereferencing code assumes that `type_byte_size` (in `util/`) word-aligns
all struct fields and inserts trailing padding so that, in well-formed
accesses, the requested data never straddles a field boundary in the
underlying primitive. Violations of that property surface as alignment
failures rather than silently corrupt encodings. The complementary
`offset_alignment` tracked in `value_sett::objectt` ensures that even
non-deterministic offsets carry an alignment lower bound, which the reference
builders use to prune cases (especially for array element accesses).

## Failed symbols

When the dereferencing code cannot produce a sensible reference — unknown
points-to, bad base type, etc. — it falls back to a fresh free symbol of the
required type, declared in the context passed to `dereferencet`'s
constructor. Names are `symex::invalid_object<N>`, where `N` is drawn from a
`thread_local` counter. The pre-existing `failed_symbol` machinery on
pointer types (queried via `dereference_callbackt::has_failed_symbol`) is
flagged as legacy in `dereference.h` and will be removed.

## Files at a glance

| File                              | Role                                                                 |
|-----------------------------------|----------------------------------------------------------------------|
| `value_set.{h,cpp}`               | Core value-set data structure, assign/get_value_set/make_union.      |
| `value_set_domain.{h,cpp}`        | Abstract domain wrapping `value_sett` for the static analyser.       |
| `value_set_analysis.{h,cpp}`      | Whole-program fixpoint over `value_set_domaint`.                     |
| `value_sets.h`                    | Abstract `value_setst` interface used by `dereferencet` clients.     |
| `dereference.{h,cpp}`             | `dereferencet`, `dereference_callbackt`, mode enum, reference builders. |
| `goto_program_dereference.{h,cpp}` | `goto_program_dereferencet`: pre-symex pass that lowers dereferences in a goto program. |
| `show_value_sets.{h,cpp}`         | `--show-value-sets` pretty-printer for debugging.                    |

## Known rough edges

* `dereference_callbackt::has_failed_symbol` is documented as legacy in
  `dereference.h`.
* `dereference_callbackt::rename` exists for future expansion and is
  currently a no-op; a FIXME in `build_reference_to` notes a benchmarking
  task before enabling it.
* `value_sett::object_numbering` is global state (per-thread). Cross-thread
  serialisation of value sets is not supported.
* The static analyser's dynamic-object handling is approximate; see the
  comments in `value_set.cpp`.
