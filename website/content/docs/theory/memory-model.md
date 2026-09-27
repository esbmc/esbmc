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
- `--malloc-zero-is-null` — let `malloc(0)` return `NULL`, as C17 7.22.3p1
  permits; the non-`NULL` alternative stays reachable, and the object it yields
  can be freed but not accessed

Every path that sizes an object — `malloc`, `calloc`, `realloc`, `alloca` and a
variable-length array declaration — is capped at `PTRDIFF_MAX`, as glibc 2.30
and later do. Above that an object's offset reads negative in the bounds checks,
in pointer subtraction and in the relational comparator, so one-past-the-end
would sort below the base. A constant request past the cap is *reported*; a
symbolic one is bounded by assumption where the path has no failure outcome to
report, and `realloc` joins the cap to its failure condition so C17 7.22.3.5
still holds. Two flags opt out: `--force-realloc-success` skips the `realloc`
cap and `--no-vla-size-check` the VLA bound.

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

A write or a `free` through a pointer the value set cannot resolve exhaustively
now fails when the pointer is none of the recorded targets, `NULL` included.
Previously `invalid_pointer(p)` answered from the integer-to-pointer
reconstruction, which enumerates only the objects registered so far, so a real
object outside the value set passed the check while symex sent the write to a
failed symbol and ran the `free` checks only for value-set targets — freed
globals, stack objects and heap interiors were missed the same way
([#7773](https://github.com/esbmc/esbmc/pull/7773)). Freeing the start of a live
allocation stays valid.

The alignment check reads the **object's base alignment** rather than assuming
the base carries the access width, so a pointer laundered out of a packed struct
no longer reads as aligned ([#7707](https://github.com/esbmc/esbmc/issues/7707)).
Where the base is below the access width the claim is decided on the whole
address instead of the offset, so a program that constrains its own object still
discharges it.

A flexible array member has size zero, as C17 6.7.2.1p18 requires. It was typed
as a one-element array, which gave the struct storage it does not have: assigning
such a struct into a heap object overwrote the bytes after the header, and an
overflow into the phantom element was missed
([#5393](https://github.com/esbmc/esbmc/issues/5393)). Dereference rebuilds
zero-width array members as empty values, which also covers GNU `[0]` members.

Memory-leak detection is enabled with `--memory-leak-check`: a dynamic object
that is still reachable-but-unfreed (or unreachable, "forgotten memory") at the
end of `main` is reported. `--no-reachable-memory-leak` keeps only the
unreachable half, and the reachability walk follows the object's contents as
well as its type, so an object held only through an allocation that was never
cast at the site — a `void *` returned by a `safe_malloc`-style wrapper, which
is modelled as a flat byte array — is not mistaken for forgotten memory.

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
