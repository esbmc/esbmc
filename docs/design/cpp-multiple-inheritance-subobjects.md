# C++ multiple-inheritance base subobjects (issues #1866, #3894)

Status: **design / in progress** — foundational analysis for reworking how the
Clang C++ frontend models base-class subobjects. Tracks esbmc/esbmc#3894
(replace the brittle `ASTRecordLayout` dependency) and fixes esbmc/esbmc#1866
(broken non-first-base access under multiple inheritance).

## 1. Symptom

```cpp
struct B { int e = 22; };
struct A { int b = 111; };
struct f : B, A {};
int main() { f i = f(); assert(i.b == 111); assert(i.e == 22); } // e wrong
```

ESBMC reports `VERIFICATION FAILED`: after construction `i.e` is
indeterminate. The correct result is `SUCCESSFUL`. Single inheritance is
unaffected (the sole base sits at offset 0).

## 2. Current model

The frontend does **not** give a derived class real nested base subobjects.
Instead `get_base_components_methods` (`clang_cpp_convert.cpp`) *flattens* every
base's components (fields, methods, and per-base `tag-Base::@vtable_pointer`)
directly into the derived `struct_typet`, deduplicated **by name**
(`is_duplicate_component`). Bases are visited through `get_base_map`, whose
`base_map` is a `std::map<std::string class_id, …>` — i.e. **alphabetical**
order, recursively merging the entire ancestry into one flat list.

Two resolution mechanisms then coexist and disagree:

* **Name-keyed** (most of the model): member access, per-base vptr selection,
  virtual dispatch, and ctor vptr-init all locate a component *by name* in the
  flattened struct.
* **Byte-offset-keyed** (bolted onto four paths): base *dtor* call, thunk
  `this`-adjust (direct bases only), the `CK_DerivedToBase` cast **when it is a
  method receiver**, and `dynamic_cast` all add
  `ASTRecordLayout::getBaseClassOffset(...)` via `(char*)p + off`.

### Why it breaks

Symex dereference through a typecast is **byte-offset** based. A base ctor call
is emitted as `Base((Base*)this)` with **no** offset
(`gen_typecast_base_ctor_call`). For a non-first base, `Base`'s first field
lives at `Base`-relative offset 0, so the write lands at derived offset 0 —
the *first* flattened component — instead of the base's real slot. Meanwhile
the derived read `i.member` is name-keyed and finds the correct component. The
write and read disagree.

Trace for the symptom (`f`'s flattened layout is alphabetical `{b, e}`):

```
B ctor:  this->e = 22   ->  { .b=22, .e=-1 }   // 22 lands in .b (offset 0)
A ctor:  this->b = 111  ->  { .b=111,.e=-1 }   // overwrites .b; .e never set
main:    i.e == 22       ->  i.e is -1  => FAILED
```

### Inventory of base-offset / `this`-adjustment sites

| Site | Location | Offset source | Status |
|---|---|---|---|
| Struct layout / member access | `get_base_components_methods`, `get_member_expr`, `build_member_from_component` | none (name flatten, alphabetical) | no offsets |
| Base **ctor** `this` | `gen_typecast_base_ctor_call` | none | **MISSING** |
| Base **dtor** `this` | `emit_base_dtor` / `build_destructor_chain` | `getBaseClassOffset` (direct) | present |
| DerivedToBase cast | `get_cast_expr` (`clang_c_convert.cpp`) | `getBaseClassOffset` over cast path | only when `is_method_receiver` (AST-shape gated — brittle, #3894) |
| BaseToDerived cast | `get_cast_expr` | none | no offset |
| Thunk `this`-adjust | `add_thunk_method_body` (`clang_cpp_convert_vft.cpp`) | `getBaseClassOffset` | **direct bases only** (indirect → 0; TODO in code) |
| General subobject offset | `offset_of_subobject` (`vft.cpp`) | `CXXBasePaths` path-sum | correct, but **only** used by `dynamic_cast` |
| Vtable type/var layout, virtual dispatch, ctor vptr-init | `vft.cpp`, `clang_cpp_convert_bind.cpp`, `clang_cpp_adjust_code_gen.cpp` | none (name-keyed) | no offset |
| Covariant returns | — | — | missing |

Consequences beyond #1866: the alphabetical flattening means ESBMC's byte
layout diverges from clang's ABI, so every `getBaseClassOffset`-based
adjustment above is only correct when a class's alphabetical base order happens
to equal its declaration order. Same-named members from two bases collapse
(`is_duplicate_component`). Indirect-base thunks and virtual bases are unhandled.

## 3. Target design

Give the derived class **real nested base subobjects**: for
`struct D : B, A`, `D`'s components are `[ @B : B, @A : A, <D's own fields> ]`
in declaration order (each `@Base` carrying that base's own layout, including
its vptr). Resolution becomes uniform and offset-free at the IR level:

* Inherited member `d.e` (from `B`) lowers to `d.@B.e` — a member path, no byte
  arithmetic.
* Base ctor/dtor/method `this` = `&d.@B` — symex computes the address from its
  own struct layout.
* Derived→base upcast = `&((D*)p)->@B`; base→derived downcast = the inverse
  member path. `dynamic_cast` walks `@Base` components.
* Per-base vptr is `d.@B.@vtable_pointer` — already distinctly named today.

This **eliminates the `ASTRecordLayout` / `getBaseClassOffset` dependency**
(the explicit ask of #3894): all offsets come from ESBMC's own layout of the
nested components, so they are self-consistent by construction, handle indirect
bases, same-named members, and (with an added virtual-base-pointer indirection)
virtual bases.

## 4. Blast radius

172 `regression/esbmc-cpp` tests use inheritance, 87 use multiple inheritance,
out of 2233 total. Every consumer of the flattened layout must move to the
subobject path: member-access lowering, `build_member_from_component`, the
cast switch, `gen_typecast_base_ctor_call`, the destructor chain, `vft.cpp`
(vptr placement + thunks), `clang_cpp_convert_bind.cpp` (dispatch),
`clang_cpp_adjust_code_gen.cpp` (vptr-init), and `dynamic_cast`.

## 5. Phased plan

Each phase keeps the full `esbmc-cpp` suite green (dual-solver where a verdict
changes) before the next.

* **P0 — analysis & harness** (this doc + a matrix of MI reproducers spanning:
  distinct-named non-first base, same-named members, 3-level indirect base,
  polymorphic MI dispatch, virtual base). *Done.*
* **P1 — nested subobject storage.** Add `@Base` components in **declaration
  order** instead of flattening; keep a compatibility shim so name-based
  lookups still resolve during migration. Validate layout via `--show-symbol-table`.
  * **P1a done:** base collection is now declaration-ordered (`base_map` is an
    ordered vector, not an alphabetical `std::map`), so the flattened layout
    agrees with clang's ABI base order. No regressions.
  * **P1b finding (byte-offset shortcut ruled out):** a trial that adjusted the
    base ctor `this` by `getBaseClassOffset` via `(char*)this + off` fixed
    simple POD MI (distinct-named `int`/DMI cases) but **regressed** polymorphic
    MI (vptr shifts the ABI offset away from the flattened layout) and, even for
    a guarded non-polymorphic case, tripped ESBMC's pointer-bounds checker when
    the base ctor *writes* a field (`inheritance11`: "Access to object out of
    bounds" in the `NetworkDevice` ctor). Conclusion: byte-offset `char*`
    arithmetic on a flattened object is not sound in ESBMC's dereference model.
    The base `this`/upcast **must** be a structural member address
    `&d.@Base` (P2), which requires the nested `@Base` storage first — it is not
    an optional optimisation but a correctness prerequisite. (The pre-existing
    dtor/method-receiver offset paths avoid the trap only because those `this`
    values are typically not written through, or are gated to narrow shapes.)
* **P2 — member access & `this` through subobjects.** Route inherited member
  access and all base ctor/dtor/method `this` through `&d.@Base`. Removes the
  ctor/dtor asymmetry and the `getBaseClassOffset` calls in those paths. Fixes
  #1866's distinct-named case.
  * **P1-core/P2 spike (validated, branch `wip/mi-nested-storage-core`):**
    `get_base_components_methods` now emits a nested `@base@<id>` component per
    base (methods still flattened as metadata); the `CK_DerivedToBase` cast is
    routed structurally (`member_exprt(d, @base@B)` for lvalues,
    `&(*p).@base@B` for pointers, chained per base-path hop); the base ctor
    `this` is routed the same way. **Result: all four #1866 reproducers pass —
    distinct-named fields, MI + own field, non-first-base method call, and the
    single-inheritance baseline — with no `char*` arithmetic and no
    pointer-bounds violations.** This confirms the structural approach is the
    correct, sound fix.
  * **Remaining migrations to reach green** (≈20 regressions, all in
    polymorphic / cast subsystems that still assume the flat layout):
    1. **vptr-init nesting (dominant):** `gen_vptr_initializations`
       (`clang_cpp_adjust_code_gen.cpp`) only iterates the derived struct's
       *direct* `is_vtptr` components, so base vptrs now living inside
       `@base@B` are never initialised → polymorphic dispatch reads a null
       vptr. Must recurse into `@base@` subobjects and emit
       `this->@base@B.@vtable_pointer = &vtable::tag-B@Derived`.
    2. **`dynamic_cast` / base→derived:** the structural downcast and
       `offset_of_subobject` path (`clang_cpp_convert_vft.cpp`) must walk
       `@base@` components instead of byte offsets.
    3. **Destructor chain:** `emit_base_dtor` still uses the byte offset;
       switch to `&this->@base@B`.
    4. **Thunks / covariant** (P4) once vptrs are nested.
  * **vptr-init migration (done):** `gen_vptr_initializations` now recurses
    into `@base@` subobjects, emitting
    `this->@base@B.@vtable_pointer = &vtable::tag-B@Derived`, so polymorphic
    dispatch through a non-first base resolves again.
  * **Virtual bases (deferred to P5):** a class with any virtual base keeps the
    legacy flattened layout (`has_virtual_bases` guard) — a shared virtual base
    must appear once in the most-derived object, which per-path nested
    subobjects cannot express yet. The cast and base-ctor `this` routing also
    self-check that the `@base@` component exists, falling back to a plain
    typecast on flattened hierarchies.
  * **Direct bases only:** nesting adds one `@base@` per *direct* base. Walking
    the transitive `base_map` duplicated ancestors (`C : B, B : A` gave `C`
    both `@base@A` and `@base@B`, the latter already nesting `@base@A`) and
    broke multi-level MI.
  * **Status: 1 failing test out of 2233** — `llbmc_multiple_inheritance`.
    **Remaining blocker (P4):** `Base2 *o = new Derived(); delete o;`. Correct
    structural upcasting makes `o` a genuine *interior* pointer
    (`&d.@base@Base2`), but `symex_free` claims "Operand of free must have zero
    pointer offset". Deleting through a base pointer with a virtual destructor
    is legal C++; the sound fix is Itanium **offset-to-top / deleting-destructor**
    semantics so the free targets the complete object. Note the pre-patch model
    only "passed" this by keeping the upcast pointer unadjusted (offset 0) and
    compensating at method-receiver casts — the stored pointer was itself wrong.
    Relaxing the zero-offset claim for `cpp_delete` would make the test pass but
    would mask genuine `delete (p+1)` bugs, so it is deliberately **not** done.
  * **Wins:** six KNOWNBUGs promoted to CORE —
    `github_1866_{distinct_names,own_field,method_call}`,
    `mi_base_subobject_layout{,_fail}`, `inheritance09`.
* **P3 — casts.** Replace the `is_method_receiver`-gated byte adjustment with
  the structural `@Base` path for all derived↔base conversions (kills the
  brittle AST-shape heuristic of #3894).
* **P4 — vtables / thunks / dispatch.** Place vptrs inside `@Base`; wire
  `offset_of_subobject` (or the structural path) into thunk `this`-adjust so
  indirect-base and non-first-base virtual dispatch are correct; add covariant
  returns.
* **P5 — virtual bases.** Shared virtual-base subobject via an indirection
  (vbase pointer), building on the #938 most-derived-init work.

## 6. Validation strategy

* Reproducer matrix under `regression/esbmc-cpp/inheritance/` (both passing and
  `_fail` variants), each cross-checked against real `clang++`/`g++` output.
* Full `esbmc-cpp` regression must stay green per phase; any KNOWNBUG that flips
  to a correct verdict is promoted to CORE with dual-solver agreement.
* No phase may rely on `--no-unwinding-assertions` with an under-unwound loop
  (masks truncation).
