---
title: Modeling with Non-determinism
weight: 10
---

ESBMC extends C with several modeling primitives for introducing
non-determinism and constraining the explored state space.

## Modeling primitives

`__ESBMC_assert(e, msg)` aborts execution when `e` is false:

```c
void __ESBMC_assert(e, "some message here");
```

`nondet_X()` returns a non-deterministic value of type `X`, where `X` is one of
`bool`, `char`, `int`, `float`, `double`, `loff_t`, `long`, `pchar`,
`pthread_t`, `sector_t`, `short`, `size_t`, `u32`, `uchar`, `uint`, `ulong`,
`unsigned`, `ushort` (no side effects). ESBMC assumes these are implemented as:

```c
X nondet_X() { X val; return val; }
```

`__ESBMC_assume(e)` ignores the current execution when `e` is false, and is a
no-op otherwise:

```c
void __ESBMC_assume(e);
```

`__ESBMC_atomic_begin()` / `__ESBMC_atomic_end()` model the atomic execution of
a sequence of statements in a multi-threaded environment:

```c
__ESBMC_atomic_begin();
// shared memory
__ESBMC_atomic_end();
```

`__ESBMC_init_object()` initializes a memory object, marking any pointer or
symbol as non-deterministic:

```c
my_complex_type T = {0, 0, 0};
__ESBMC_init_object(T);
```

## Example

The following program uses non-determinism to search for a Pythagorean triple:

```c
int main() {
  int x = nondet_int(), y = nondet_int(), z = nondet_int();
  __ESBMC_assume(x > 0 && y > 0 && z > 0);
  __ESBMC_assume(x < 16384 && y < 16384 && z < 16384);
  assert(x*x + y*y != z*z);
  return 0;
}
```

Invoke ESBMC with `esbmc file.c`, and it produces a counterexample:

```
Counterexample:

State 1 file file.c line 2 function main thread 0
----------------------------------------------------
x = 252 (00000000 00000000 00000000 11111100)

State 2 file file.c line 2 function main thread 0
----------------------------------------------------
y = 561 (00000000 00000000 00000010 00110001)

State 3 file file.c line 2 function main thread 0
----------------------------------------------------
z = 615 (00000000 00000000 00000010 01100111)

State 6 file file.c line 5 function main thread 0
----------------------------------------------------
Violated property:
file file.c line 5 function main
assertion
(_Bool)(x * x + y * y != z * z)

VERIFICATION FAILED
```
