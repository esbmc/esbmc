---
title: Modeling with Non-determinism
weight: 10
---

ESBMC extends C with several modeling primitives for introducing
non-determinism and constraining the explored state space.

## Modeling primitives

`__ESBMC_assert(cond, msg)` reports a property violation when `cond` is false:

```c
void __ESBMC_assert(_Bool cond, const char *msg);

__ESBMC_assert(x > 0, "x must be positive");
```

`nondet_X()` returns a non-deterministic, side-effect-free value of type `X`.
ESBMC forward-declares a convenience set where `X` is one of `bool`, `char`,
`schar`, `uchar`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `float`,
`double`, conceptually defined as:

```c
X nondet_X() { X val; return val; }
```

The same functions are also available under the SV-COMP `__VERIFIER_nondet_X`
spelling. More generally, **any function whose body is unavailable returns a
fresh non-deterministic value of its return type**, so an external function that
is declared but not defined behaves like a `nondet_` call.

`__ESBMC_assume(e)` ignores the current execution when `e` is false, and is a
no-op otherwise (also available as `__VERIFIER_assume`):

```c
void __ESBMC_assume(_Bool e);
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
