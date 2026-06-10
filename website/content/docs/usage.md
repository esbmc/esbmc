---
title: Usage
weight: 2
---

As an illustrative example to show some of the ESBMC features concerning floating-point numbers, consider the following C code:

```c
#include <assert.h>	
#include <math.h>
unsigned char nondet_uchar();
double nondet_double();
int main() {
unsigned char N = nondet_char();
double x = nondet_double();

if(x <= 0 || isnan(x))
  return 0;

unsigned short i = 0;

__VERIFIER_assume(x < 5);

double x_0 = x;  // Store initial value

__VERIFIER_assume(N >= 0 && N < 2);

// Loop invariant: 0 ≤ i ≤ N and x = x_0 * 2^i and x > 0
while(i < N) {
  __VERIFIER_assume(x > 0);
  __VERIFIER_assume(x == x_0 * pow(2, i));
  __VERIFIER_assume(0 <= i && i <= N);

  x = (2 * x);
  ++i;
}

assert(x > 0);
  return 0;
}
```

Here, ESBMC is invoked as follows: `esbmc file.c --floatbv --k-induction` where `file.c` is the C program to be checked, `--floatbv` indicates that ESBMC will use floating-point arithmetic to represent the program's `float` and `double` variables, and `--k-induction` selects the k-induction proof rule. The user can select the SMT solver, property, and verification strategy. For this particular C program, ESBMC provides the following output as the verification result:

```
*** Checking inductive step
Starting Bounded Model Checking
Unwinding loop 2 iteration 1 file ex5.c line 8 function main
Not unwinding loop 2 iteration 2 file ex5.c line 8 function main
Symex completed in: 0.001s (40 assignments)
Slicing time: 0.000s (removed 16 assignments)
Generated 2 VCC(s), 2 remaining after simplification (24 assignments)
No solver specified; defaulting to Bitwuzla
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.005s
Solving with solver Bitwuzla
Encoding to solver time: 0.005s
Runtime decision procedure: 0.427s
BMC program time: 0.435s

VERIFICATION SUCCESSFUL

Solution found by the inductive step (k = 2)
```

As an illustrative example to show some of the ESBMC features concerning pointer safety, consider the following C code:

```c
#include <stdlib.h>
int *a, *b;
int n;
#define BLOCK_SIZE 128
void foo () {
  int i;
  for (i = 0; i < n; i++)
    a[i] = -1;
  for (i = 0; i < BLOCK_SIZE - 1; i++)
    b[i] = -1;
}
int main () {
  n = BLOCK_SIZE;
  a = malloc (n * sizeof(*a));
  b = malloc (n * sizeof(*b));
  *b++ = 0;
  foo ();
  if (b[-1]) { free(a); free(b); }
  else { free(a); free(b); }
  return 0;
}
```

Here, ESBMC is invoked as follows:

```sh
esbmc file.c --memory-leak-check
```

where `file.c` is the C program to be checked and `--memory-leak-check`
indicates that ESBMC will check for memory leaks. For this particular C program,
ESBMC produces the following counterexample:

```
Counterexample:

State 1 file ex2.c line 14 function main thread 0
----------------------------------------------------
a = (signed int *)(&dynamic_1_array[0])

State 2 file ex2.c line 15 function main thread 0
----------------------------------------------------
b = (signed int *)0

State 3 file ex2.c line 16 function main thread 0
----------------------------------------------------
b = 0 + 1

State 6 file ex2.c line 16 function main thread 0
----------------------------------------------------
Violated property:
  file ex2.c line 16 function main
  dereference failure: NULL pointer
```

In the counterexample shown above, State 1 indicates that memory has been allocated, as indicated by 'dynamic_1_array'. State 2 indicates that the `malloc` call failed and returned NULL, indicating that the memory was not allocated. Note that ESBMC allows the user to skip checking for malloc/new failures via `--force-malloc-success`. State 3 represents an assignment to pointer b. Lastly, State 6 reports a failure to dereference pointer b.

As an illustrative example to show some of the ESBMC features concerning concurrency, consider the following C code:

```c
#include <assert.h>	
#include <pthread.h>
int n=0; //shared variable
pthread_mutex_t mutex;
void* P(void* arg) {
  int tmp, i=1;
  while (i<=10) {
    pthread_mutex_lock(&mutex);
    tmp = n;
    n = tmp + 1;
    pthread_mutex_unlock(&mutex);
    i++;
  }
  return NULL;
}
int main (void) {
  pthread_t id1, id2;
  pthread_mutex_init(&mutex, NULL);
  pthread_create(&id1, NULL, P, NULL);
  pthread_create(&id2, NULL, P, NULL);
  pthread_join(id1, NULL);
  pthread_join(id2, NULL);
  assert(n == 20);
}
```

Here, we create two threads `id1` and `id1`; both threads will run the same code as implemented in **P**. Note that these two threads communicate via the shared memory `n`, which is protected by a mutex via **pthread_mutex_lock** and **pthread_mutex_unlock**. Note further that the thread `main` contains two joining points via **pthread_join** for `id1` and `id2`.

ESBMC can be invoked as follows: `esbmc file.c --context-bound 2` where `file.c` is the C program to be checked, and `--context-bound nr` limits the number of context switches for each thread. For this particular C program, ESBMC produces the following verification result:

```
*** Thread interleavings 612 ***
Unwinding loop 1 iteration 10 file test3.c line 6 function P
Unwinding loop 1 iteration 1 file test3.c line 6 function P
Unwinding loop 1 iteration 2 file test3.c line 6 function P
Unwinding loop 1 iteration 3 file test3.c line 6 function P
Unwinding loop 1 iteration 4 file test3.c line 6 function P
Unwinding loop 1 iteration 5 file test3.c line 6 function P
Unwinding loop 1 iteration 6 file test3.c line 6 function P
Unwinding loop 1 iteration 7 file test3.c line 6 function P
Unwinding loop 1 iteration 8 file test3.c line 6 function P
Unwinding loop 1 iteration 9 file test3.c line 6 function P
Unwinding loop 1 iteration 10 file test3.c line 6 function P
Symex completed in: 0.031s (431 assignments)
Slicing time: 0.001s (removed 183 assignments)
Generated 149 VCC(s), 7 remaining after simplification (248 assignments)
No solver specified; defaulting to Bitwuzla
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.004s
Solving with solver Bitwuzla
Encoding to solver time: 0.004s
Runtime decision procedure: 0.001s
BMC program time: 0.040s

VERIFICATION SUCCESSFUL
```

## Verifying Python Programs

ESBMC has a dedicated Python frontend. See the [Python](/docs/python) section for
how to verify Python programs, the supported features, and worked examples.

## Witness Generation

When ESBMC refutes a property, it produces a counterexample that can be used to debug the program to find the root cause of the problem. For this purpose, ESBMC can produce the counterexample in graphml format to make its evaluation easier (e.g., by building a tool that allows graphical visualization).

As an illustrative example, consider the following fragment of C code, where we declare two bit-vectors of size 10 each: x and y, and then check whether x == y.

```c
#include <assert.h>

int main() {
  _ExtInt(10) x = nondet_float();
  _ExtInt(10) y = nondet_int();
  assert(x == y);
  return 0;
}
```

If we call ESBMC as `esbmc main.c --witness-output main.graphml`, where `main.c` is the C program we want to verify, while `main.graphml` stores the counterexample in graphml format, then ESBMC will produce the following output:

```sh
esbmc main.c --witness-output main.graphml
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <key id="frontier" attr.name="isFrontierNode" attr.type="boolean" for="node">
    <default>false</default>
  </key>
  <key id="violation" attr.name="isViolationNode" attr.type="boolean" for="node">
    <default>false</default>
  </key>
  <key id="entry" attr.name="isEntryNode" attr.type="boolean" for="node">
    <default>false</default>
  </key>
  <key id="sink" attr.name="isSinkNode" attr.type="boolean" for="node">
    <default>false</default>
  </key>
  <key id="cyclehead" attr.name="cyclehead" attr.type="boolean" for="node">
    <default>false</default>
  </key>
  <key id="sourcecodelang" attr.name="sourcecodeLanguage" attr.type="string" for="graph"/>
  <key id="programfile" attr.name="programfile" attr.type="string" for="graph"/>
  <key id="programhash" attr.name="programhash" attr.type="string" for="graph"/>
  <key id="creationtime" attr.name="creationtime" attr.type="string" for="graph"/>
  <key id="specification" attr.name="specification" attr.type="string" for="graph"/>
  <key id="architecture" attr.name="architecture" attr.type="string" for="graph"/>
  <key id="producer" attr.name="producer" attr.type="string" for="graph"/>
  <key id="sourcecode" attr.name="sourcecode" attr.type="string" for="edge"/>
  <key id="startline" attr.name="startline" attr.type="int" for="edge"/>
  <key id="startoffset" attr.name="startoffset" attr.type="int" for="edge"/>
  <key id="control" attr.name="control" attr.type="string" for="edge"/>
  <key id="invariant" attr.name="invariant" attr.type="string" for="node"/>
  <key id="invariant.scope" attr.name="invariant.scope" attr.type="string" for="node"/>
  <key id="assumption" attr.name="assumption" attr.type="string" for="edge"/>
  <key id="assumption.scope" attr.name="assumption" attr.type="string" for="edge"/>
  <key id="assumption.resultfunction" attr.name="assumption.resultfunction" attr.type="string" for="edge"/>
  <key id="enterFunction" attr.name="enterFunction" attr.type="string" for="edge"/>
  <key id="returnFromFunction" attr.name="returnFromFunction" attr.type="string" for="edge"/>
  <key id="endline" attr.name="endline" attr.type="int" for="edge"/>
  <key id="endoffset" attr.name="endoffset" attr.type="int" for="edge"/>
  <key id="threadId" attr.name="threadId" attr.type="string" for="edge"/>
  <key id="createThread" attr.name="createThread" attr.type="string" for="edge"/>
  <key id="witness-type" attr.name="witness-type" attr.type="string" for="graph"/>
  <graph edgedefault="directed">
    <data key="producer">ESBMC 6.7.0</data>
    <data key="sourcecodelang">C</data>
    <data key="architecture">64bit</data>
    <data key="programfile">main.c</data>
    <data key="programhash">7ba149c407ef7ae9e971bbc937b37a624575d6a5</data>
    <data key="specification">CHECK( init(main()), LTL(G ! call(__VERIFIER_error())) )</data>
    <data key="creationtime">2021-06-07T13:37:38</data>
    <data key="witness-type">violation_witness</data>
    <node id="N0">
      <data key="entry">true</data>
    </node>
    <node id="N1"/>
    <edge id="E0" source="N0" target="N1">
      <data key="enterFunction">main</data>
      <data key="createThread">0</data>
    </edge>
    <node id="N2"/>
    <edge id="E1" source="N1" target="N2">
      <data key="startline">4</data>
      <data key="assumption">x = -512;</data>
      <data key="threadId">0</data>
    </edge>
    <node id="N3"/>
    <edge id="E2" source="N2" target="N3">
      <data key="startline">5</data>
      <data key="assumption">y = -166;</data>
      <data key="threadId">0</data>
    </edge>
    <node id="N4">
      <data key="violation">true</data>
    </node>
    <edge id="E3" source="N3" target="N4">
      <data key="startline">93</data>
      <data key="threadId">0</data>
    </edge>
  </graph>
</graphml>
```
    
We recommend reading [Exchange Format for Violation Witnesses and Correctness Witnesses](https://github.com/sosy-lab/sv-witnesses) to obtain further information about violation and correctness witnesses in graphml format.

## Unwinding Assertions

In ESBMC, all loops are "unwound", i.e., replaced by several guarded copies of the loop body; the same happens for backward "gotos" and recursive functions. Soundness requires that ESBMC insert a so-called `unwinding assertion` at the end of the loop. As an example, consider the simple C code fragment illustrated below:

```c
unsigned int x=∗;
while ( x>0) x−−;
assert ( x==0);
```

Note that the loop in line 2 runs an unknown number of times, depending on the initial non-deterministic value assigned to x in line 1. The assertion in line 3 holds independently of x's initial value. BMC tools typically fail to verify programs that contain such loops. In particular, BMC tools introduce an unwinding assertion at the end of the loop, as illustrated in line 5 of this C code fragment.

```c
unsigned int x=∗;
if(x>0)
  x−−;   // k copies
  ...
assert (!(x>0));
assert(x==0);
```

This unwinding assertion in line 5 causes the BMC tool to fail if _k_ is too small, as follows:

```c
#include <assert.h>
unsigned int nondet_uint();
int main() {
  unsigned int x=nondet_uint();
  while(x>0) x--;

  assert(x==0);
  return 0;
}
```

```sh
esbmc file.c --unwind 3
```

```
Counterexample:

State 1 file file.c line 4 function main thread 0
----------------------------------------------------
x = 3170305 (00000000 00110000 01100000 00000001)

State 2 file file.c line 5 function main thread 0
----------------------------------------------------
x = 3170304 (00000000 00110000 01100000 00000000)

State 3 file file.c line 5 function main thread 0
----------------------------------------------------
x = 3170303 (00000000 00110000 01011111 11111111)

State 4 file file.c line 5 function main thread 0
----------------------------------------------------
Violated property:
file file.c line 5 function main
unwinding assertion loop
```

## Verification Strategies

ESBMC offers several incremental strategies that control how loops are unwound
and whether correctness can be proven. For an in-depth explanation of how each
algorithm works, see
[Verification Algorithms](/docs/theory/verification-algorithms).

| Flag | Strategy | Proves correctness? |
|---|---|---|
| `--falsification` | Iteratively unwind, looking only for bugs | No (bug-finding only) |
| `--incremental-bmc` | Iteratively unwind; also detect full unrolling | Yes, once all loops fully unwind |
| `--k-induction` | Base case + forward condition + inductive step | Yes |

`--max-k-step N` caps the unwind bound (default 50); `--k-step N` changes the
increment granularity.

## Verifying modules that span multiple files

ESBMC can verify code that relies on existing infrastructure. Consider a program
whose `mul` function lives in a separate library:

```c
#include "lib.h"
// Running with: esbmc --overflow-check main.c lib.c
int main() {
  int64_t a, b, r;
  if (mul(a, b, &r)) {
    __ESBMC_assert(r == a * b, "Expected result from multiplication");
  }
  return 0;
}
```

Invoke ESBMC with the include path and the implementation file:

```sh
esbmc main.c --overflow-check -I lib/ lib/lib.c
```

where `--overflow-check` enables arithmetic over-/underflow checks and `-I path`
sets the include path. The library under `lib/` is:

```c
// lib.h
#include <stdint.h>
_Bool mul(const int64_t a, const int64_t b, int64_t *res);
```

```c
// lib.c
#include "lib.h"
_Bool mul(int64_t a, int64_t b, int64_t *res) {
  if ((a == 0) || (b == 0)) { *res = 0; return 1; }
  else if (a == 1)          { *res = b; return 1; }
  else if (b == 1)          { *res = a; return 1; }
  *res = a * b;   // there exists an overflow
  return 1;
}
```

ESBMC reports the overflow at the unguarded multiplication:

```
Counterexample:

State 1 file lib.c line 14 function mul thread 0
----------------------------------------------------
Violated property:
file lib.c line 14 function mul
arithmetic overflow on mul
!overflow("*", a, b)

VERIFICATION FAILED
```

## Multiple Property Verification

```sh
esbmc file.c --multi-property
```

ESBMC can verify the satisfiability of all claims of a given bound. In
multi-property mode, ESBMC does not stop at the first counterexample; it
continues until all bugs are found. Relevant options:

- `--multi-property` — verify all claims of the current bound (also activates `--no-remove-unreachable`).
- `--multi-fail-fast N` — stop after the first `N` violations.
- `--keep-verified-claims` — do not skip verified claims (assertions inside a loop body are then re-verified during unwinding).
- `--all-witnesses` — after a property is violated, enumerate further inputs that also violate it (implies `--multi-property`; see below).
- `--max-witnesses N` — cap witnesses per property (default 16; 0 = unlimited).

### Enumerating all violating inputs

```sh
esbmc file.c --all-witnesses
esbmc file.c --all-witnesses --max-witnesses 4
```

By default `--multi-property` reports a single counterexample per failing
property. `--all-witnesses` instead enumerates *distinct concrete input vectors*
that violate the same property, until the set is exhausted (UNSAT) or the
`--max-witnesses` cap is reached — useful for fault localisation, test-case
mining, and characterising the failing-input sub-domain.

```c
#include <assert.h>
int main(void) {
  int x;                 // nondet
  if (x > 0) x--; else x++;
  assert(x != 0);        // violated by x == 1 AND x == -1
  return 0;
}
```

`esbmc file.c --all-witnesses` reports both witnesses:

```
[Counterexamples – 2 witnesses]

  Witness 1 of 2
    Inputs : [0] = -1
  Witness 2 of 2
    Inputs : [0] = 1

Summary: 2 distinct input tuples violate this property
         (enumeration stopped: UNSAT after 2 witnesses)
```

Internally the same SMT instance is re-solved with a blocking clause over the
nondet input symbols, so enumerating *N* witnesses is much cheaper than running
ESBMC *N* times. Floating-point inputs are handled specially (the NaN
equivalence class is excluded as a whole; other values use bit-pattern equality,
so `+0` and `-0` are distinct). The footer states why enumeration stopped; only
*UNSAT after N* means the witness set is complete.

**Implementation notes.** Blocking clauses are scoped to a single SMT context
frame (`push_ctx`/`pop_ctx`) per claim, so the feature is safe under
`--smt-during-symex` and does not leak between claims. Machine-readable artifacts
(`--cex-output`, `--generate-testcase`, `--generate-html-report`,
`--generate-json-report`, `--witness-output-graphml`, `--witness-output-yaml`)
fan out per witness using the `<ce>-<file>` prefix scheme, one file per witness,
so it is also safe under `--parallel-solving`. Enumeration is skipped during the
inductive step of k-induction (a SAT result there means UNKNOWN, not a real
counterexample).

**Scaling caveat.** The textual report dumps each witness's full goto trace; with
`--no-slice` on a deeply unrolled program this grows quickly. If you only need the
violating inputs, the per-witness `Inputs : ...` line is usually enough, and the
machine-readable per-witness files give the full data without the noise.

Formally this is *bounded projected model enumeration* for a fixed property: the
blocking-clause loop from SAT all-solutions algorithms, lifted to SMT and
projected onto the nondet input symbols. See McMillan, *Applying SAT Methods in
Unbounded Symbolic Model Checking*, CAV 2002
([doi](https://doi.org/10.1007/3-540-45657-0_19)), and Grumberg, Schuster,
Yadgar, *Memory Efficient All-Solutions SAT Solver and Its Application for
Reachability Analysis*, FMCAD 2004
([doi](https://doi.org/10.1007/978-3-540-30494-4_20)). It complements
dynamic-symbolic-execution test generation (KLEE, DART, SAGE), which varies the
path condition to maximise coverage; `--all-witnesses` instead fixes the failure
path and enumerates input vectors on it.

## Supported SMT backends {#smt-backends}

ESBMC integrates several SMT solvers directly via their APIs, and on Unix can
also drive an external solver process over a pipe:

| Backend | Option |
|---|---|
| Bitwuzla | `--bitwuzla` (default) |
| Boolector | `--boolector` |
| Z3 | `--z3` |
| MathSAT | `--mathsat` |
| CVC4 | `--cvc` |
| Yices | `--yices` |
| SMTLIB | `--smtlib --smtlib-solver-prog CMD` |

An alternative default solver can be set with `--default-solver SOLVER` (the
name without the `--`), which suits a shell alias or the `ESBMC_OPTS`
environment variable. The `CMD` for the SMTLIB backend is interpreted by the
shell, so it can include options or chain commands (the tools must be on
`PATH`):

- `boolector --incremental`
- `z3 -in`
- `tee formula.smt2 | z3 -in | tee output.txt`
- `yices-smt2 --incremental`
- `cvc5 -L smt2 -m`

Remember to quote the `CMD` string when invoking ESBMC.
