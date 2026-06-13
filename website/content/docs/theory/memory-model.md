---
title: Memory Model and Pointer Safety
weight: 15
---

ESBMC verifies pointer-manipulating programs by giving every pointer a precise
symbolic meaning and checking each dereference against the state of the memory
it refers to [1]. This page explains how pointers are modelled and which safety
properties ESBMC derives from that model.

## Pointers as object + offset

ESBMC does not model a pointer as a flat machine address. Instead, each pointer
value is a pair:

- an **object** — which allocated entity (a variable, an array, a heap block,
  or the special *invalid* / *NULL* object) the pointer refers to, and
- an **offset** — the byte displacement into that object.

Two intrinsics expose the two components and are used internally throughout the
encoding:

```c
unsigned __ESBMC_POINTER_OBJECT(const void *p);  // which object p points at
signed   __ESBMC_POINTER_OFFSET(const void *p);  // byte offset into that object
```

Pointer arithmetic moves the offset while keeping the object fixed, so
`p + i` and `p` always share an object. This object/offset split is what lets
ESBMC reason about spatial safety symbolically: a dereference is in bounds iff
its offset lies within the size of its object.

## Dynamic allocation and lifetimes

Heap allocation (`malloc`, `calloc`, `realloc`, C++ `new`) creates a fresh
dynamic object. ESBMC tracks each object's allocation state in internal
bookkeeping (the `__ESBMC_alloc` map and the `__ESBMC_is_dynamic` predicate),
so it knows at every program point whether an object is live, already freed, or
never allocated. `free`/`delete` mark the object deallocated; a later access
through a pointer to it is then a use-after-free.

By default ESBMC also explores the possibility that an allocation *fails*
(returns `NULL`), so code that dereferences the result without checking is
flagged. This can be tuned:

- `--force-malloc-success` — assume allocation never fails
- `--malloc-zero-is-null` — model `malloc(0)` as returning `NULL`

## Properties checked

From the model above, ESBMC derives the standard spatial and temporal
memory-safety properties. The relevant checks are on by default (the flags below
*disable* them) except memory-leak detection, which is opt-in:

| Property | Disable with |
|---|---|
| Array / buffer bounds | `--no-bounds-check` |
| Pointer dereference validity (NULL, invalid, out-of-object, use-after-free) | `--no-pointer-check` |
| Pointer alignment | `--no-align-check` |
| Relational comparison of pointers into different objects | `--no-pointer-relation-check` |

`free`-specific diagnostics include freeing a pointer with a non-zero offset
("Operand of free must have zero pointer offset"), freeing an invalid or
already-freed pointer ("invalid pointer freed", double free), and freeing
non-dynamic storage.

Memory-leak detection is enabled with `--memory-leak-check`: a dynamic object
that is still reachable-but-unfreed (or unreachable, "forgotten memory") at the
end of `main` is reported.

## Why formulas are pointer-heavy

Because every dereference carries an object identity and an offset, and because
ESBMC models the lifetimes of all dynamically allocated objects, a large part of
a generated [SMT formula](/docs/theory/smt-formula-generation) is pointer-safety
bookkeeping rather than program arithmetic — see *About Formula Size* on the SMT
page.

## References

[1] Lucas C. Cordeiro, Bernd Fischer, João Marques-Silva: *SMT-Based Bounded
Model Checking for Embedded ANSI-C Software.* IEEE Trans. Software Eng.
38(4):957–974, 2012. The paper describes ESBMC's SMT encoding of pointers,
arrays, structures, and unions. [doi:10.1109/TSE.2011.59](https://doi.org/10.1109/TSE.2011.59)
