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

## Architecture at a glance

```text
  goto program                                       SMT formula
       |                                                  ^
       |   (one of two value-set providers)               |
       v                                                  |
  +---------------------+                                 |
  | value_setst         |  abstract interface             |
  | (value_sets.h)      |  used by dereferencet           |
  +---------------------+                                 |
   ^                    ^                                 |
   |                    |                                 |
   |                    |                                 |
  +-------------------+ +----------------------------+    |
  | value_set_        | | goto_symex's callback impl |    |
  | analysist         | | (uses live per-state       |    |
  | (pre-symex        | |  value_sett)               |    |
  |  fixpoint over    | +----------------------------+    |
  |  value_set_       |          |                        |
  |  domaint)         |          | get_value_set(...)     |
  +-------------------+          v                        |
            |              +-------------------+          |
            |              | dereferencet      |          |
            +------------->|                   |          |
                           |  dereference_expr |  -> dispatch on expr_id
                           |    |  |  |        |
                           |    v  v  v        |
                           |  guard / addrof / |
                           |  nonscalar chain  |
                           |    |              |
                           |    v              |
                           |  dereference()    |  -> per-candidate if-chain
                           |    |              |
                           |    v              |
                           |  build_reference_to  -> alignment/bounds/valid
                           |    |              |
                           |    v              |
                           |  build_reference_rec -> src/dst/offset switch
                           |    |              |
                           |    v              |
                           |  construct_from_* / stitch_together_from_byte_array
                           +-------------------+
                                    |
                                    v
                          expr2tc fed to SMT backend
```

`value_setst` is the indirection that lets `dereferencet` consume points-to
information without caring whether it came from the static analyser or from
the live `value_sett` carried by a goto-symex state.

## Key invariants

These hold throughout the pipeline; the rest of the document assumes them.

* **Bit-offset currency in the dereferencer.** Inside
  `build_reference_to` / `build_reference_rec` the offset is in **bits**, not
  bytes. The byte → bit conversion happens once at the top of
  `build_reference_to` (`mul ... 8`); the lexical chain in
  `dereference_expr_nonscalar` also accumulates bits via
  `compute_pointer_offset_bits`.
* **Word-aligned struct layout.** `type_byte_size` (in `util/`) word-aligns
  every struct field and inserts trailing padding. A well-formed scalar
  access therefore never straddles a field boundary in the SMT primitive;
  anything that would surfaces as an alignment failure rather than silently
  corrupt the encoding.
* **Alignment lower bound on non-deterministic offsets.**
  `value_sett::objectt::offset_alignment` is the minimum byte alignment of a
  symbolic offset (1 if nothing better is known). The reference builders use
  it to prune dispatch cases — most importantly, array-element accesses stay
  on the clean per-element path instead of falling through to byte stitching.
* **Failed symbols are well-typed.** When the dereferencer cannot produce a
  sensible reference (unknown points-to, type mismatch, NULL in
  read/write mode), it returns a fresh free symbol
  `symex::invalid_object<N>` of the requested type. The SMT formula stays
  well-formed; the corresponding `dereference_failure` assertion is what
  the user is meant to notice.
* **Thread-locality of identifier state.** `value_sett::object_numbering`,
  `obj_numbering_refset`, and `dereferencet::invalid_counter` are
  `thread_local` so parallel symex (`--k-induction-parallel`) does not race
  on shared numbering or symbol names.

## Formal model

The two services above have a compact formal reading. This section gives (a) the
*grammar* of the points-to facts the analysis manipulates, and (b) the *transfer
functions* that map program expressions and assignments onto those facts. Both
are exactly what `value_set.cpp` computes; the function or branch that realises
each rule is named alongside it.

### Grammar of points-to facts

A value-set state `ρ` (the `value_sett::values` map) is a set of entries, each
binding a *pointer location* to the set of objects it may point at. In EBNF
(`|` alternation, `{ }` zero-or-more, `[ ]` optional):

```text
state      ::= { entry }                          (* value_sett::values *)
entry      ::= location "=" "{" [ object { "," object } ] "}"
location   ::= identifier suffix                  (* entryt.identifier ++ suffix *)
identifier ::= l1_name                            (* an l1 variable *)
             | "value_set::dynamic_object" int    (* heap block *)
             | "value_set::return_value"
suffix     ::= { "." field | "[]" }               (* struct member / array-of-ptr *)

object     ::= "<" referent "," offset "," align "," type ">"
             | "unknown"                          (* unknown2t: may point anywhere *)
             | "invalid"                          (* invalid2t: provably bad      *)
referent   ::= l1_name | "dynamic_object" int | "NULL"
offset     ::= int | "*"                          (* "*" = symbolic; offset_is_set=false *)
align      ::= int                                (* bytes; >= 1 whenever offset = "*"    *)
```

This is the syntax `value_sett::output` prints under `--show-value-sets`
(`value_set.cpp:22-108`): every `object` is a `<referent, offset, align, type>`
tuple, except `unknown`/`invalid` referents, which print bare. Internally each
tuple is an `objectt` (`offset`, `offset_is_set`, `offset_alignment`) keyed, in
the entry's `object_mapt`, by an index into `object_numbering` that names the
referent. `align` is `objectt::offset_alignment` in bytes and is meaningful
precisely when `offset` is symbolic.

When a value set is *exported* to a client (`get_value_set` → `valuest`,
`value_set.cpp:182-194`) each tuple becomes an `object_descriptor2t(referent,
offset, align)` expression, while `unknown`/`invalid` stay as `unknown2t` /
`invalid2t`. That `valuest` is exactly what `dereferencet` consumes.

### Transfer functions

Write `ρ ⊢ e ⇓ M` for "in state `ρ`, expression `e` evaluates to the object-map
`M`" — the relation computed by `value_sett::get_value_set_rec`. Each rule is
tagged with the branch that implements it:

```text
                ρ(p · suffix) = M
[SYM]    ---------------------------------                is_symbol2t
                  ρ ⊢ p ⇓ M

[NULL]   ρ ⊢ NULL ⇓ { <NULL, 0, ..> }                    is_symbol2t, thename = "NULL"

             refset(x) = M
[ADDR]   ----------------------                          is_address_of2t → get_reference_set
             ρ ⊢ &x ⇓ M

           ρ ⊢ a ⇓ Ma     ρ ⊢ b ⇓ Mb
[IF]     ------------------------------                  is_if2t
           ρ ⊢ (c ? a : b) ⇓ Ma ∪ Mb

             ρ ⊢ e ⇓ M
[CAST]   ------------------                              is_typecast2t / is_bitcast2t
           ρ ⊢ (T)e ⇓ M

           ρ ⊢ p ⇓ M     k constant,  s = sizeof(*p)
[PTR+k]  ----------------------------------------------  is_add2t/is_sub2t, constant arm
           ρ ⊢ p + k ⇓ { <o, off + k·s, a> : <o, off, a> ∈ M, off ≠ * }

           ρ ⊢ p ⇓ M     i not constant
[PTR+i]  ----------------------------------------------  is_add2t/is_sub2t, nondet arm
           ρ ⊢ p + i ⇓ { <o, *, min(nat(o), align(p))> : <o, _, _> ∈ M }

           refset(*p) = R     ∀ o ∈ R.  ρ ⊢ o ⇓ M_o
[DEREF]  ----------------------------------------------  is_dereference2t
           ρ ⊢ *p ⇓  ⋃ M_o

[NEW]    ρ ⊢ malloc/new @ ℓ ⇓ { <dynamic_object_ℓ, 0, 1> }   is_sideeffect2t (alloc)

             ρ ⊢ e ⇓ M
[MEMB]   --------------------------------                is_member2t
           ρ ⊢ e.f ⇓ M  with suffix ".f"

             ρ ⊢ e ⇓ M
[INDEX]  --------------------------------                is_index2t
           ρ ⊢ e[i] ⇓ M  with suffix "[]"

[OTHER]  ρ ⊢ e ⇓ { unknown }   (any expr none of the above match)
```

Two properties are worth stressing, both visible in the rules:

* **Arrays are summarised, not enumerated.** `[INDEX]` attaches the suffix `[]`
  rather than tracking element `i`, so a whole array of pointers shares one
  entry. This is what makes the analysis terminate on unbounded arrays, at the
  cost of per-element precision.
* **Pointer arithmetic degrades the offset, never the object set.** `[PTR+k]`
  keeps the referent set identical and only shifts offsets; `[PTR+i]` keeps the
  set but widens every offset to symbolic with the alignment lower bound
  `min(nat(o), align(p))`. The `offset_alignment` machinery described under
  *Key invariants* is exactly this `align` component.

### Assignment and join

An assignment rebinds one location to the value set of its right-hand side
(`value_sett::assign`, `value_set.cpp:1074`):

```text
            ρ ⊢ rhs ⇓ M
[ASSIGN] ------------------------------------------------
          ρ ⊢ (lhs = rhs) ⇒ ρ[ lhs ↦ M ]               (symex: overwrite)
                          ⇒ ρ[ lhs ↦ ρ(lhs) ⊔ M ]      (static: join, add_to_sets=true)
```

The join `⊔` (`make_union` / `insert`, `value_set.cpp:168`, `value_set.h:350`)
is set union on referents; when the *same* referent carries two offsets it
widens conservatively:

```text
   <o, n, _>  ⊔  <o, n, _>  =  <o, n>                          (equal: unchanged)
   <o, a, _>  ⊔  <o, b, _>  =  <o, *, min(align(a), align(b))> (a ≠ b)
   <o, *, x>  ⊔  <o, _, y>  =  <o, *, min(x, y)>               (either symbolic)
```

So two definite-but-different offsets for one object collapse to a symbolic
offset with the coarser alignment — the lattice climbs from "exact offset" to
"aligned-but-unknown offset" and finally (via `unknown`) to "points anywhere."

### Worked examples

Each line shows the resulting entry in `--show-value-sets` notation
(`<referent, offset, align, type>`; `*` = symbolic offset). The `align` column
is written abstractly as `a` (a byte alignment lower bound): its concrete value
follows the `offset_alignment` rules and the 8-byte scalar default in
`get_natural_alignment`, and is best read off `--show-value-sets` for a given
program. Only the referent set and the offset are pinned down here:

```text
int g;
int *p = &g;            p = { <g, 0, a, signedbv> }       [ADDR]
int *q = p;             q = { <g, 0, a, signedbv> }       [SYM]
p = p + 3;              p = { <g, 12, a, signedbv> }      [PTR+k] (offset += 3·sizeof(int))
p = p + nondet_int();   p = { <g, *, a', signedbv> }      [PTR+i] (offset widened; a' <= a)

struct S { int *fld; } s;
s.fld = &g;             s.fld = { <g, 0, a, signedbv> }   [MEMB]  (suffix ".fld")

int *a[10];
a[i] = &g;              a[] = { <g, 0, a, signedbv> }     [INDEX] (one entry, all i)

int *h = malloc(n);     h = { <dynamic_object_ℓ, 0, a, ...> }   [NEW]  (ℓ = alloc site)

int *r = c ? &g : &h;   r = { <g, 0, a, signedbv>,
                              <h, 0, a, signedbv> }       [IF]
```

The longer `*baz = 1` walkthrough later in this document
([Example walkthrough](#example-walkthrough)) shows `[IF]`, `[PTR+i]`, and the
join interacting within a single statement.

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

## Example walkthrough

Take the snippet from the intro:

```c
int foo, bar;
char *baz = (nondet_bool()) ? (char *)&foo : (char *)&bar;
unsigned int idx = nondet_uint();
__ESBMC_assume(idx < sizeof(int));
baz += idx;
*baz = 1;
```

The discussion stops at the `expr2tc` tree handed to the SMT backend — the
actual SMT lowering depends on the chosen solver theory (bitvector vs.
floatbv, `concat`/`extract` vs. `select`/`store`) and is best inspected with
`esbmc --show-vcc` against the program in question.

### After `baz = (...) ? &foo : &bar`

`value_sett::assign(baz, rhs)` walks the conditional and unions both arms.
The resulting entry for `baz` (suffix empty, since `baz` itself is the
pointer) has two object map keys, one for `foo` and one for `bar`. Both
arms carry a constant zero offset:

```text
baz = { <foo, 0, 1, signedbv>, <bar, 0, 1, signedbv> }
```

`<obj, offset, alignment, type>` is the format `value_sett::output` prints
under `--show-value-sets` (see `value_set.cpp:60-88`). `offset` is `*` when
symbolic; `alignment` is the byte alignment lower bound from
`objectt::offset_alignment`.

### After `baz += idx`

`baz += idx` is interpreted by `value_sett::get_value_set_rec` for the
right-hand side; the `add` over a pointer with a non-constant operand
collapses both entries' constant offsets into symbolic ones, with
`offset_alignment` recomputed via `offset2align`. Since the base objects
are `int` (4 bytes natural alignment) but the arithmetic is on a `char *`,
the alignment lower bound drops to 1:

```text
baz = { <foo, *, 1, signedbv>, <bar, *, 1, signedbv> }
```

### Dispatch of `*baz = 1`

`*baz` reaches `dereference_expr` with `expr_id == dereference_id`, which
recurses into the operand under `READ` mode and then calls
`dereference(baz, char, guard, WRITE, expr2tc())`. There is no lexical
member/index chain, so `extra_offset` is nil.

`dereference` queries the callback for the points-to set and iterates over
the two candidates. For each, `build_reference_to`:

1. runs `check_pointer_alignment` (passes — `char` requires no alignment);
2. builds `same_object2tc(baz, &foo)` as the candidate pointer guard;
3. computes the final offset: the value-set entry has `offset_is_set ==
   false`, so the offset becomes `pointer_offset(baz)`, then × 8 for bits;
4. calls `valid_check`, then `check_data_obj_access` (the destination is
   an `int`-typed SMT primitive being written via a `char` projection);
5. calls `build_reference_rec(foo, 8*pointer_offset(baz), char, guard,
   WRITE, 8)`.

In `build_reference_rec`, `value->type` is `int` (scalar), `type` is `char`
(scalar), and the offset is not a `constant_int2t`, so the flags resolve to
`flag_src_scalar | flag_dst_scalar | flag_is_dyn_offs`. That case dispatches
to `construct_from_dyn_offset`, which goes through
`stitch_together_from_byte_array`: the `int` primitive is decomposed into
four byte expressions via `extract_bytes`, and the chosen byte is selected
by the symbolic offset modulo the width. The resulting `expr2tc` is the
projection of one byte out of `foo`.

The same construction runs for `bar`. `dereference` chains the two results:

```text
if(same_object(baz, &bar),
   <byte projection of bar via pointer_offset(baz)>,
   <byte projection of foo via pointer_offset(baz)>)
```

For a write, the symex layer turns this rvalue selection into a
`with`-style update on each candidate primitive, guarded by the same
`same_object2tc` predicates plus the surrounding control-flow guard.

### Assertions emitted along the way

The `__ESBMC_assume(idx < sizeof(int))` keeps `pointer_offset(baz)` within
range, so the `bounds_check` calls inside `build_reference_to` discharge
trivially under the solver. Without the assumption, the same code path
still runs but the bounds assertion becomes reachable and reportable.

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
