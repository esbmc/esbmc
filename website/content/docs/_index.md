---
title: Documentation
next: /docs/setup
---

* A set of slides about the detection of software vulnerabilities using ESBMC:
  * [Part I](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/slides/lecture03.pdf)
  * [Part II](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/slides/lecture04.pdf)
  * [Part III](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/slides/lecture05.pdf)
* This [software security](https://ssvlab.github.io/lucasccordeiro/courses/2022/01/software-security/index.html) course describes further implementation details about ESBMC.


## ESBMC Features

<ul>
  <li>User specified assertion failures;</li>
  <li>Out of bounds array access;</li>
  <li>Illegal pointer dereferences, such as:
  <ul>
  <li>Dereferencing null;</li>
  <li>Performing an out-of-bounds dereference;</li>
  <li>Double-free of malloc'd memory;</li>
  <li>Misaligned memory access;</li>
  </ul></li>
  <li>Integer overflows;</li>
  <li>NaN (Floating-point);</li>
  <li>Divide by zero;</li>
  <li>Memory leak.</li>
</ul>

<p>Concurrent software (using the pthread API) is verified by explicitly exploring interleavings, thus producing one symbolic execution per interleaving. By default, pointer-safety, array-out-of-bounds, division-by-zero, and user-specified assertions  will be checked for; one can also specify options to check multi-threaded programs for:</p>

<ul>
<li>Deadlock (only on pthread mutexes and conditional variables);</li>
<li>Data races (<i>i.e.</i>, competing writes);</li>
<li>Atomicity violations at visible assignments;</li>
<li>Lock acquisition ordering.</li>
</ul>

<p>By default, ESBMC performs a "lazy" depth-first search of interleavings but can also encode (explicitly) all interleavings into a single SMT formula. Currently, many SMT solvers are supported:</p>

<ul>
<li>Z3 4.9+;</li>
<li>Boolector 3.2+;</li>
<li>MathSAT 5.6+;</li>
<li>CVC4 1.8;</li>
<li>Yices 2.6+;</li>
<li>Bitwuzla 0.3+;</li>
</ul>

<p>In addition, ESBMC can be configured to use the SMTLIB interactive text format to write the formula to a file or interactively with a pipe to communicate with an arbitrary solver process, although insignificant overheads are involved.
See the section on <a href="#smt-backends">supported SMT backends</a> for details.</p>

<p>ESBMC uses clang as its front-end, which brings several advantages:</p>

<ul>
<li>We address the problem of maintaining a frontend for C and C++ simply and elegantly: by using clangs API to access and traverse the program AST, without having details of the input program compiled away.</li>
<li>ESBMC provides compilation error messages as expected from a compiler.</li>
<li>ESBMC leverages clang’s powerful static analyzer to provide meaningful warnings when parsing the program.</li>
<li>Clang can simplify some expressions, e.g., calculate <i>sizeof/alignof</i> expressions, evaluate static asserts, evaluate if a dynamic cast is always null, etc., which eases the analysis of the input program.</li>
</ul>
<p>A limited subset of C++98 is supported too -- a library modeling the STL is also available.</p>

<p>To check all available options of the ESBMC tool, type:</p>

```sh
esbmc --help
```

## Modeling with non-determinism

<p>ESBMC extends C with three modeling features:</p>

<p> __ESBMC_assert(e): aborts execution when <i>e</i> is false. </p>

```c
void __ESBMC_assert (e, "some message here");

```

<script src="js/shCore.js"></script>
<script src="js/shBrushCpp.js"></script>
<script>
SyntaxHighlighter.all()
</script>

<p> nondet_X(): returns non-deterministic X-value, with X in {bool, char, int, float, double, loff_t, long, pchar, pthread_t, sector_t, short, size_t, u32, uchar, uint, ulong, unsigned, ushort} (no side effects, pointer for void *, etc.). ESBMC assumes that the functions are implemented according to the following template:</p>

```c
X nondet_X () { X val; return val; }
```

<script src="js/shCore.js"></script>
<script src="js/shBrushCpp.js"></script>
<script>
SyntaxHighlighter.all()
</script>

`__ESBMC_assume(e):` "ignores" execution when `e` is false, no-op otherwise.

```c
void __ESBMC_assume(e);
```

`__ESBMC_atomic_begin()`, `__ESBMC_atomic_end()`: For modeling an atomic execution of a sequence of statements in a multi-threaded run-time environment, those statements can be placed between two function calls.

```c
__ESBMC_atomic_begin();
//shared memory
__ESBMC_atomic_end();

```

<script src="js/shCore.js"></script>
<script src="js/shBrushCpp.js"></script>
<script>
SyntaxHighlighter.all()
</script>

<p> __ESBMC_init_object(): Initialize a memory object. This can be used to mark any pointer or symbol as non-determnistic.</p>

```c
my_complex_type T = {0,0,0};
__ESBMC_init_object(T);
```

<p>As an illustrative example to show some of the ESBMC features to model non-determinism, consider the following C code:  </p>

```c
int main() {
  int x=nondet_int(),y=nondet_int(),z=nondet_int();
  __ESBMC_assume(x > 0 && y > 0 && z > 0);
  __ESBMC_assume(x < 16384 && y < 16384 && z < 16384);
  assert(x*x + y*y != z*z);
  return 0;
}
```

<p>Here, ESBMC is invoked as follows:</p>

```sh
esbmc file.c
```

<p>For this particular C program, ESBMC produces the following counterexample:</p>


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


## Quantifiers

<p>ESBMC now supports universal (<code>forall</code>) and existential (<code>exists</code>) quantifiers in SMT-based verification.</p>

### New Expressions

<p>The following two expressions have been introduced:</p>
<ul>
  <li><code>bool forall(symbol, predicate)</code> - Evaluates if the predicate holds for all possible values of <code>symbol</code>.</li>
  <li><code>bool exists(symbol, predicate)</code> - Evaluates if the predicate holds for at least one value of <code>symbol</code>.</li>
</ul>

### Function Declarations

```c
extern void __ESBMC_assume(_Bool);
extern _Bool __ESBMC_forall(void *, _Bool);
extern _Bool __ESBMC_exists(void *, _Bool);
```

### Example Usage

```c
int main() {
unsigned n;
int arr[n];
unsigned i;

__ESBMC_assume(__ESBMC_forall(&i, !(i < n) || arr[i] == 2));
__ESBMC_assert(!__ESBMC_exists(&i, (i < n) && arr[i] == 42), "forall init");

arr[n/2] = 42;
__ESBMC_assert(!__ESBMC_exists(&i, (i < n) && arr[i] == 42), "this assertion should fail");
}
```
    
### Another Example

```c
int zero_array[10];
int main() {
int sym;
__ESBMC_assert(
  __ESBMC_forall(&sym, !(sym >= 0 && sym < 10) || zero_array[sym] == 0),
  "array is zero initialized"
);

const unsigned N = 10;
unsigned i = 0;
char c[N];
for (i = 0; i < N; ++i) c[i] = i;

unsigned j;
__ESBMC_assert(
  __ESBMC_forall(&j, j > 9 || c[j] == j), "array is initialized correctly"
);
}
```

```sh
esbmc file.c --z3
```

### Limitations

<ul>
  <li>Currently, only Z3 is supported (no SMT-LIB support).</li>
  <li>Only one symbol is supported in quantifiers. Future work will enable multiple symbols.</li>
  <li>Recursive quantifiers (e.g., nested <code>forall</code> statements) are not yet supported.</li>
  <li>There is a known issue where a constant-bounded symbol might cause incorrect simplifications.</li>
</ul>

### Falsification

<h3 id="falsification"></h3>

```sh
esbmc file.c --falsification
```

<p>Our falsification approach (<i>--falsification</i>) uses an iterative technique and verifies the program for each unwind bound up to either a maximum default value of <i>50</i> (which can be changed via --max-k-step nr), or indefinitely (until it exhausts the time or memory limits). Intuitively, we aim to find a counterexample with up to <i>k</i> loop unwindings. The algorithm relies on the symbolic execution engine to increasingly unwind the loop after each iteration.</p>

<p>This approach replaces all unwinding assertions (e.g., assertions to check if a loop was completely unrolled) with unwinding assumptions. Normally, this would lead to unsound behaviour but, since the falsification algorithm cannot provide correctness validation, it will not affect the search for bugs. This approach is focused on bug finding and does not care if a loop was not completely unrolled; it only cares if the current number of unwindings will lead to a property violation.</p>

<p>The falsification algorithm also offers the option to change the granularity of the increment; the default value is <i>1</i>, but can be increased in order to meet any expected behaviour via --k-step nr. Note that changing the value of the increment can lead to slower verification time and might not present the shortest counterexample possible for a property violation.</p>

<h3 id="incremental-bmc">Incremental BMC</h3>

```sh
esbmc file.c --incremental-bmc
```

<p>Our incremental BMC approach (<i>--incremental-bmc</i>) uses an iterative technique and verifies the program for each unwind bound up to either a maximum default value of <i>50</i>, which can be modified via --max-k-step nr, or indefinitely (until it exhausts the time or memory limits). Intuitively, we aim to either find a counterexample with up to <i>k</i> loop unwinding or to fully unwind all loops so we can provide a correct result. The algorithm relies on the symbolic execution engine to increasingly unwind the loop after each iteration of the algorithm.</p>

<p>The approach is divided in two steps: one that tries to find property violations and one that checks if all the loops were fully unwound. When searching for property violation, the tool replaces all unwinding assertions (e.g., assertions to check if a loop was completely unrolled) with unwinding
assumptions. Normally, this would lead to unsound behaviour, however, the first step can only find property violations and reporting an unwinding assertion failure is not a real bug. The next step is to check if all loops in the program were fully unrolled. This is done by checking if all the unwinding assertions are unsatisfiable; note that checking any other assertion in the program, for the current <i>k</i>, is not necessary as they were already verified. </p>

<p>The algorithm also offers the option to change the granularity of the increment; the default value is <i>1</i>, but can be increased in order to meet any expected behaviour via --k-step nr. Note that changing the value of the increment can lead to slower verification time and might not present the shortest counterexample possible for the property violation.</p>

<h3 id="k-induction">k-Induction proof rule</h3>

```sh
esbmc file.c --k-induction
```

<p>The original <i>k</i>-induction algorithm (<i>--k-induction</i>) presented by Sheeran et al. [1] was used to prove safety properties in hardware verification. The algorithm was later refined by Alastair et al. [2] and applied to the verification of general C programs. Our algorithm is a combination of both approaches. It can be summarized as follows:</p>

<div style="text-align: center; margin: 20px 0;">
  <div style="border: 1px solid #ccc; padding: 15px; display: inline-block;">
    <div>¬ <i>B(k)</i> → program contains bug</div>
    <div><i>B(k)</i> ∧ <i>F(k)</i> → program is correct</div>
    <div><i>B(k)</i> ∧ <i>I(k)</i> → program is correct</div>
  </div>
</div>

<p>Here <i>B(k)</i> is the base case, <i>F(k)</i> is the forward condition and <i>I(k)</i> is the inductive step; <i>k</i> is the number of loop unwinding used for each step. For the base case we use the plain BMC technique, hence we can only find property violations here. If the base case error check is satisfiable, then the algorithm presents a counterexample of length <i>k</i>. For the forward condition and inductive step, the base case must be checked for satisfiability before the result is presented. This is a soundness requirement of the technique.</p>

<p>The forward condition attempts to prove that all loops in the program were fully unrolled; this is achieved by adding <i>unwinding assertions</i> after all loops. The forward condition is further optimized to only check the <i>unwinding assertions</i>, as all program assertions are already proven to be unsatisfiable by the base case, for the current value of <i>k</i>. The inductive step attempts to prove that, if the property is valid for <i>k</i> iterations, then it must be valid for the next iteration; this is achieved by assigning nondeterministic values to all variable written inside a loop body, assuming <i>k-1</i> invariants and checking if the invariant holds at the <i>k</i>th iteration.</p>

<p>The algorithm starts with <i>k = 1</i>. It increases it up to a maximum number of iterations, incrementally analysing the program, until it either finds a bug (i.e., the base case is satisfiable for some <i>k</i>), proves correctness (i.e., the base case is unsatisfiable and either the forward condition or inductive step is unsatisfiable for some <i>k</i>), or exhausts either time or memory constraints.</p>

<h4 id="loop-invariants">Loop Invariant Support</h4>

```sh
esbmc file.c --loop-invariant
```


<p>ESBMC now supports user-provided loop invariants as an alternative to expensive loop unwinding. This approach is particularly beneficial for programs with large loop bounds or unbounded loops, where traditional k-induction may become computationally prohibitive or hit iteration limits.</p>

<p>Loop invariants are specified using the built-in function <code>__ESBMC_loop_invariant(condition)</code> placed within the loop body. When the <i>--loop-invariant</i> option is enabled, ESBMC transforms the loop using the standard k-induction approach:</p>

<ol>
  <li><strong>Base case verification:</strong> Check that the invariant holds upon loop entry</li>
  <li><strong>Inductive step:</strong> Assume the invariant holds, execute one loop iteration, and verify the invariant still holds</li>
  <li><strong>Property verification:</strong> After loop exit, use the invariant to prove the target property</li>
</ol>

#### Example: Basic Loop Invariant Usage</h5>

```c
int main() {
int i = 0;
int sum = 0;

__ESBMC_loop_invariant(i >= 0 && i <= 1000 && sum == i * 10);
while (i < 1000) {
  sum += 10;
  i++;
}

assert(sum == 10000);  // Successfully verified
  return 0;
}
```

<p>The loop invariant approach has been tested on various benchmark programs from the k-induction test suite, successfully verifying several cases that previously resulted in "VERIFICATION UNKNOWN" verdicts due to unwinding limitations, including <code>check_if</code>, <code>count_down</code>, and <code>count_up_down</code>.</p>

## Known Limitations

<div class="warning">
  <p><strong>Integer Overflow False Positives:</strong> The current implementation may generate false positive overflow errors due to aggressive havoc operations that create values outside expected bounds. This occurs when nondeterministic values are assigned without proper constraint propagation.</p>

  <p><strong>Nested Loop Support:</strong> The current implementation does not correctly handle nested loops with multiple invariants. State management between inner and outer loops requires further refinement.</p>

  <p><strong>Manual Invariant Specification:</strong> Users must manually specify correct loop invariants since ESBMC will not validate or infer them before verifying the program's correctness.</p>
</div>

<p>We can also use additional options together with the k-induction proof rule to produce (inductive) invariants:</p>
<ol>
  <li><i>--interval-analysis</i>: enable interval analysis for integer variables and add assumes to the program.</li>
  <li><i>--add-symex-value-sets</i>: enable value set analysis for pointers and add assumes to the program.</li>
  <li><i>--loop-invariant</i>: enable user-provided loop invariants to avoid expensive loop unwinding.</li>
</ol>

<p>The loop invariant feature is completely opt-in and maintains backward compatibility with existing verification workflows. Programs without loop invariant annotations continue to use the standard k-induction unwinding approach.</p>

<p>[1] Mary Sheeran, Satnam Singh, Gunnar Stålmarck: Checking Safety Properties Using Induction and a SAT-Solver. FMCAD 2000: 108-125</p>
<p>[2] Alastair F. Donaldson, Leopold Haller, Daniel Kroening, Philipp Rümmer: Software Verification Using k-Induction. SAS 2011: 351-368
  
<h3 id="multiple-files">Verification of modules that rely on larger structures</h3>

<p>ESBMC can verify code that relies on existing infrastructures and must be compliant with those. Consider the following C program where the verification engineer wants to check whether the assert-statement in line 8 holds.</p>


```

1 #include "lib.h"
2 // Running with esbmc  --overflow-check main.c lib.c
3 int main() {
4   int64_t a;
5   int64_t b;
6   int64_t r;
7   if (mul(a, b, &r)) {
8   __ESBMC_assert(r == a * b, "Expected result from multiplication");
9   }
10   return 0;
11 }

```

<script src="js/shCore.js"></script>
<script src="js/shBrushCpp.js"></script>
<script>
SyntaxHighlighter.all()
</script>

The function <i>mul</i> is implemented in the library "lib.h", which is located under "/lib". Here, ESBMC is invoked as follows:

```sh
esbmc main.c --overflow-check -I lib/ lib/lib.c
```

<p>where <i>main.c</i> is the C program to be checked, <i>--overflow-check</i> enables arithmetic over- and underflow check, and <i>-I path</i> sets the include path. For this particular C program, ESBMC produces the following counterexample: </p>


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


<p>The library header and implementation files located under <i>/lib</i> are:</p>

    
```

1 #include <stdint.h>
2 _Bool mul(const int64_t a, const int64_t b, int64_t *res);

```

<script src="js/shCore.js"></script>
<script src="js/shBrushCpp.js"></script>
<script>
SyntaxHighlighter.all()
</script>


```

1 #include "lib.h"
2 _Bool mul(int64_t a, int64_t b, int64_t *res) {
3   // Trivial cases
4   if((a == 0) || (b == 0)) {
5   *res = 0;
6   return 1;
7   } else if(a == 1) {
8   *res = b;
9   return 1;
10   } else if(b == 1) {
11   *res = a;
12   return 1;
13   }
14   *res = a * b; // there exists an overflow
15   return 1;
16 }
```

<h3 id="esbmc-python">Verification of Python Programs</h3>

<p> To enable the verification of Python programs, build ESBMC with the option:
</br><code>'-DENABLE_PYTHON_FRONTEND=On'</code>.</p>

<p>Users can specify the Python interpreter binary using a flag.</p>
```sh
esbmc --help
```

```
Python frontend:
--python path             Python interpreter binary to use 
                  (searched in $PATH; default: python)
```


```sh
esbmc main.py --python python2.7
```

```
ESBMC version 7.6.1 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py
Python version: 2.7.18
ERROR: Please ensure Python 3 is available in your environment.
```


```sh
esbmc main.py --python python3
```

```
ESBMC version 7.6.1 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing main.py
Converting
Loading model: range.py
Loading model: int.py
Generating GOTO Program
GOTO program creation time: 0.151s
GOTO program processing time: 0.002s
Starting Bounded Model Checking
Symex completed in: 0.002s (14 assignments)
Slicing time: 0.000s (removed 10 assignments)
Generated 9 VCC(s), 2 remaining after simplification (4 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.000s
Solving with solver Boolector 3.2.3
Runtime decision procedure: 0.000s
BMC program time: 0.003s

VERIFICATION SUCCESSFUL
```

<p> Consider the following Python file, named main.py: </p>

```
def factorial(n:int) -> int:
if n == 0 or n == 1:
  return 1
else:
  return n * factorial(n - 1)

n:int = nondet_int()
__ESBMC_assume(n > 0);
__ESBMC_assume(n < 6);

result:int = factorial(n)
assert(result != 120)
```


<p> Run ESBMC on the Python file using the following command:</p>
```c
$ esbmc main.py --incremental-bmc 
```

<p>ESBMC will analyze the program and detect the assertion violated when 'factorial' is invoked with the value 5. The counterexample generated by the tool will demonstrate this:</p>


```
[Counterexample]


State 1 file main.py line 7 column 0 thread 0
----------------------------------------------------
n = 5 (00000000 00000000 00000000 00000101)

State 4  thread 0
----------------------------------------------------
Violated property:
assertion
result != 120


VERIFICATION FAILED

Bug found (k = 5)
```


<p>The <code>--function</code> flag can be used to verify a single function instead of the entire file:</p>
```c
$ esbmc <python-file> --function <function-name>
```

<p>This command instructs ESBMC to focus only on the specified function, making the verification process more efficient when you are only interested in a particular part of the code.</p>


<h3 id="esbmc-solidity">Verification of Solidity Smart Contracts</h3>

<p> ESBMC has a frontend to process Solidity source code and hence can verify simple Solidity smart contracts. In order to verify Solidity smart contract, ESBMC should be built with the option <b>'-DENABLE_SOLIDITY_FRONTEND=On'</b>.</p>
<p> There are three relevant options, which are:</p>
<ul>
<li><b>sol:</b> sets the smart contract source file (<b>.sol</b> and <b>.solast</b>)</li>
<li><b>contract:</b> sets the target contract name</li>
<li><b>function:</b> sets the target function name</li>
</ul>
<p> As an illustrative example, consider the following Solidity code: </p>


```

1 // SPDX-License-Identifier: GPL-3.0
2 pragma solidity >=0.4.26;
3
4 contract MyContract {
5
6   function func_array_loop() external pure {
7   uint8[2] memory a;
8
9   a[0] = 100;
10  for (uint8 i = 1; i < 3; ++i)
11  {
12    a[i] = 100;
13    assert(a[i-1] == 100);
14  }
15  }
16 }


```

<script src="js/shCore.js"></script>
<script src="js/shBrushCpp.js"></script>
<script>
SyntaxHighlighter.all()
</script>

<p> As declared in line 7, <i>a</i> is an static array of the size 2. The loop in line 10 will try to write 10 in <i>a[2]</i> in the third iteration, which is out-of-bound access. This error can be detected by ESBMC using the command lines as follows: </p>

```sh
esbmc --sol example.sol example.solast --contract MyContract --function func_array_loop --incremental-bmc
```

<p> where <i>MyContract.solast</i> is the JSON AST of the Solidity source code generated using the command line below:</p>

```sh
solc --ast-compact-json example.sol > example.solast
```

<p> Since there is no ambiguous function name, the <b>--contract</b> option can be omitted. Note that the solidity compiler version should be greater or equal than 0.4.26. For this example, ESBMC produces the following counterexample:</p>


```

Counterexample:

State 1 file example.sol line 1 function func_array_loop thread 0
----------------------------------------------------
a[0] = 100 (01100100)

State 2 file example.sol line 1 function func_array_loop thread 0
----------------------------------------------------
a[1] = 100 (01100100)

State 4 file example.sol line 1 function func_array_loop thread 0
----------------------------------------------------
Violated property:
file example.sol line 1 function func_array_loop
array bounds violated: array `a' upper bound
(signed long int)i < 2


VERIFICATION FAILED

Bug found (k = 2)

```


<p>Like other state-of-art verifiers, ESBMC can also verify state properties. A common type of properties in smart contracts are properties that involve the state of the contract. Multiple transactions might be needed to make an assertion fail for such a property. Consider a a 2D grid: </p>


```

pragma solidity >=0.8.0;

contract Robot {
int x = 0;
int y = 0;

function moveLeftUp() public {
  --x;
  ++y;
}

function moveLeftDown() public {
  --x;
  --y;
}

function moveRightUp() public {
  ++x;
  ++y;
}

function moveRightDown() public {
  ++x;
  --y;
}

function inv() public view {
  assert((x + y) % 2 != 0);
}
}

```


ESBMC prove that the assertion(Invariant) could be violated throughout the funtion calls, via command:

```sh
esbmc --sol example.sol example.solast --contract Robot --k-induction
```

<p>The counterexample shows the path that leads to the assertion failure:</p>

```
[Counterexample]


State 1 file example.sol line 13 function moveLeftDown thread 0
----------------------------------------------------
x = -2 (11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111110)

State 2 file example.sol line 14 function moveLeftDown thread 0
----------------------------------------------------
y = 0 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 3 file example.sol line 18 function moveRightUp thread 0
----------------------------------------------------
x = -1 (11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111)

State 4 file example.sol line 19 function moveRightUp thread 0
----------------------------------------------------
y = 1 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000001)

State 5 file example.sol line 23 function moveRightDown thread 0
----------------------------------------------------
x = 0 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 6 file example.sol line 24 function moveRightDown thread 0
----------------------------------------------------
y = 0 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 7 file example.sol line 28 function inv thread 0
----------------------------------------------------
Violated property:
file example.sol line 28 function inv
assertion
(x + y) % 2 != 0


VERIFICATION FAILED

```


<p>We provide a technical report about the verification of Solidity programs <a href="https://arxiv.org/pdf/2111.13117.pdf">here</a>.</p>
<p>We also provide a Github action for security verification of solidity contracts using ESBMC-solidity <a href="https://github.com/actions-marketplace-validations/alanpjohn_esbmc-solidity-action" target=”_blank”>here</a>.</p>

<h3 id="multiple-property-verification">Multiple Property Verification</h3>

```sh
esbmc file.c --multi-property
```

<p>

ESBMC can verify the satisfiability of all the claims of a given bound. During this multi-property verification, ESBMC does not terminate when a counterexample is found; instead, it continues to run until all bugs have been discovered. There are three relevant options, which are:</p>
<ul>
<li><b>multi-property:</b> verifies the satisfiability of all claims of
the current bound. This also activates <b>--no-remove-unreachable</b>.</li>
<li><b>multi-fail-fast n:</b> stops after first <b>n</b> VCC violation found in multi-property mode</li>
<li><b>keep-verified-claims:</b> do not skip verified claims in multi-property verification. With this option enabled, assertions inside the loop body will be verified repeatedly during the unwinding; while with this option disabled, the claims will only get verified once.</li>
</ul>
<p>An example of multi-property verification can be found in the <b>Code  Coverage Metric</b> section below.</p>

<h3 id="code-coverage-metric">Code Coverage Metric</h3>
<p>
ESBMC provides a set of coverage metrics to help you measure how much of the state space you've visited. The supported coverage metrics can be listed as follows:</p>
<ul>
<li><b>Assertion Coverage</b> measures how well the assertions within a program are tested.</li>
<li><b>Condition Coverage</b> measures how well the Boolean expressions in the code have been tested.</li>
<li><b>Branch Coverage</b> measures how well every possible branch (or path) in a decision point of the code has been executed.</li>
  </ul>

  <p>
  As an illustrative example, consider the following C code:
  </p>

```c
int main()
{
  int x = 0;
  while (nondet_int()) {
    if (!x) {
      assert(x == 0);
      x = 1;
    }
    else if (x == 1) {
      assert(x > 0);
      x = 2;
    }
    else if (x == 2) {
      assert(x >= 2);
      x = 3;
    }
  }
  assert(x == 3);
}
```


<p>For assertion coverage, ESBMC is invoked as follows:</p>
```sh
esbmc example.c --k-induction --assertion-coverage
```

<p>For this particular C program, ESBMC produces the following counterexample and coverage information, reflecting that two branches in the program leave unexplored during verification:</p>

```
[Counterexample]


State 1 file example.c line 24 column 3 function main thread 0
----------------------------------------------------
Violated property:
file example.c line 24 column 3 function main
x == 3
0

Slicing time: 0.000s (removed 0 assignments)
No solver specified; defaulting to Boolector
Solving claim 'x == 0' with solver Boolector 3.2.2
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.001s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.000s

[Counterexample]


State 1 file example.c line 10 column 7 function main thread 0
----------------------------------------------------
Violated property:
file example.c line 10 column 7 function main
x == 0
0


[Coverage]

Total Asserts: 4
Total Assertion Instances: 4
Reached Assertion Instances: 2
Assertion Instances Coverage: 50%

VERIFICATION FAILED

Bug found (k = 1)

```


<p>
<ul>
  <li><b>Total Asserts:</b> the total number of assertions that are contained in the flow that ESBMC covers.</li>
  <li><b>Total Assertion Instances:</b> the number of times that assertion can be triggered after ESBMC folds the code. For example, if a loop with 4 iterations contains an assertion, this assertion has 4 instances</li>
  <li><b>Reached Assertion Instances:</b> the number of verified assertion instances. By using <b>--condition-coverage-claims</b>, the guard and location information of the instances are also listed</li>
  <li>The <b>coverage</b> is obtained by dividing reached assertion instances by total assertion instances.</li>
  <li>The <b>unreached claims</b> can be checked by comparing them with the output of <b>--show-claims</b>.</li>
</ul>
</p>
<p>For condition coverage, ESBMC is invoked as follows:</p>
```sh
esbmc example.c --k-induction --condition-coverage-claims
```
<p>The output coverage result can be illustrated as follows:</p>

```
[Coverage]

!((_Bool)return_value$_nondet_int$1 != 0)   file example.c line 6 column 3 function main : SATISFIED
(_Bool)return_value$_nondet_int$1 != 0    file example.c line 6 column 3 function main : SATISFIED
!((_Bool)x != 0)    file example.c line 8 column 5 function main : SATISFIED
(_Bool)x != 0 file example.c line 8 column 5 function main : SATISFIED
!(x == 1)   file example.c line 13 column 10 function main : SATISFIED
x == 1    file example.c line 13 column 10 function main : SATISFIED
!(x == 2)   file example.c line 18 column 10 function main : SATISFIED
x == 2    file example.c line 18 column 10 function main : SATISFIED
!(x == 3)   file example.c line 24 column 3 function main : SATISFIED
x == 3    file example.c line 24 column 3 function main : SATISFIED
x == 0    file example.c line 10 column 7 function main : SATISFIED
!(x == 0)   file example.c line 10 column 7 function main : UNSATISFIED
x > 0 file example.c line 15 column 7 function main : SATISFIED
!(x > 0)    file example.c line 15 column 7 function main : UNSATISFIED
x >= 2    file example.c line 20 column 7 function main : SATISFIED
!(x >= 2)   file example.c line 20 column 7 function main : UNSATISFIED
Reached Conditions:  16
Short Circuited Conditions:  0
Total Conditions:  16

Condition Properties - SATISFIED:  13
Condition Properties - UNSATISFIED:  3

Condition Coverage: 81.25%

VERIFICATION FAILED
```

<p>
<ul>
<li><b>Total Conditions:</b> the total number of Boolean conditions</li>
<li><b>Short Circuited Conditions:</b> the number of conditions that are short-circuited. This refers to the conditions in Boolean expressions that are not eventually evaluated as soon as the result is determined</li>
<li><b>Reached Conditions:</b> the total number of conditions that are reached during the verification</li>
<li><b>Condition Properties - SATISFIED/UNSATISFIED:</b> the number of conditions that are satisfied/unsatisfied</li>
<li><b>Condition Coverage:</b> is obtained by dividing reached assertion instances by total assertion instances.</li>
</ul>
</p>

<p>
Note that the <b>--condition-coverage-claims</b> option provides verbose output of claim information, including its condition and location. If only the coverage number is needed, we recommend using the <b>--condition-coverage</b> option instead.
</p>

<h3 id="smt-backends">Supported SMT backends</h3>
<p>ESBMC integrates a number of SMT solvers directly via their respective
  API, but on Unix can also be instructed to communicate with an external
  SMT solver process by a pipe. The following table lists ESBMC's options
  enabling the use of the particular solver.</p>

<table>
<thead>
  <tr><td>Backend</td><td>Option</td></tr>
</thead>
<tr><td>Boolector</td><td><code>--boolector</code> (this is the default)</td></tr>
<tr><td>Z3</td><td><code>--z3</code></td></tr>
<tr><td>MathSAT</td><td><code>--mathsat</code></td></tr>
<tr><td>CVC4</td><td><code>--cvc</code></td></tr>
<tr><td>Yices</td><td><code>--yices</code></td></tr>
<tr><td>Bitwuzla</td><td><code>--bitwuzla</code></td></tr>
<tr><td>SMTLIB</td><td><code>--smtlib --smtlib-solver-prog CMD</code>
  (see below for details about the placeholder <code>CMD</code>)</td></tr>
</table>

<p>While Boolector is the default, an alternative default solver can also
  be specified with the <code>--default-solver SOLVER</code> option, where
  <code>SOLVER</code> corresponds to one of the above options without the
  <code>--</code>. This option is particular suited for a shell alias or the
  <code>ESBMC_OPTS</code> environment variable, which is parsed every time
  ESBMC runs.</p>

<p>The <code>CMD</code> parameter for the SMTLIB backend is a string that is
  interpreted by the shell, therefore it can contain additional options
  to a particular command separated by whitespace, or even chain together
  multiple commands. Here are some examples for CMD that work with ESBMC.
  Note that the tools in these commands are assumed to be available
  through the <code>PATH</code> environment variable:</p>
<ul>
<li><code>boolector --incremental</code></li>
<li><code>z3 -in</code></li>
<li><code>tee formula.smt2 | z3 -in | tee output.txt</code></li>
<li><code>yices-smt2 --incremental</code></li>
<li><code>cvc5 -L smt2 -m</code></li>
</ul>

<p>Remember to quote the <code>CMD</code> string when executing ESBMC.</p>

<h3 id="esbmc-support">ESBMC Support</h3>

<p>We are still increasing the robustness of ESBMC and also continuously implementing new features, more optimizations and experiencing new encodings. For any question about ESBMC, please contact us via <code>https://github.com/esbmc/esbmc</code>.</p>
