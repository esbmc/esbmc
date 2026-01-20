---
title: Counterexample Guided Abstract Refinement Integer vs Bit-Vector Encoding
---

### Example 1: Checking the Pythagorean Theorem

Consider the following C program that resembles the Pythagorean theorem in mathematics:

````C
#include <assert.h>

int main() {
  _BitInt(100000) a = nondet_int();
  _BitInt(100000) b = nondet_int();
  _BitInt(100000) c = nondet_int();

  __ESBMC_assume(a > 0 && b > 0 && c > 0);

  assert((a * a) + (b * b) == (c * c));
}
````

This program verifies whether the values of `a`, `b`, and `c` satisfy the equation: `a^2 + b^2 = c^2`.

If we use bit-vector encoding, most state-of-the-art SMT solvers struggle to find a counterexample that refutes this assertion due to the large integers (i.e., _BitInt(100000)). However, if we use integer encoding, the solver can quickly find a counterexample, as shown below:

````
[Counterexample]

State 1 file example1.c line 4 column 3 function main thread 0
----------------------------------------------------
  a = 1

State 2 file example1.c line 5 column 3 function main thread 0
----------------------------------------------------
  b = 1

State 3 file example1.c line 6 column 3 function main thread 0
----------------------------------------------------
  c = 2

State 4 file example1.c line 10 column 3 function main thread 0
----------------------------------------------------
Violated property:
  file example1.c line 10 column 3 function main
  assertion (a * a) + (b * b) == (c * c)
  a * a + b * b == c * c


VERIFICATION FAILED
````

We can produce an executable test case to confirm this assertion violation as follows:

````
$ clang example1-test.c -o example1-test
$ ./example1-test
example1-test: example1-test.c:9: int main(): Assertion `(a * a) + (b * b) == (c * c)' failed.
Aborted (core dumped)
````

### Key Takeaways:

* Bit-vector encoding makes it harder for SMT solvers to find counterexamples due to the bit-precise arithmetic at large bit-widths.
* Integer encoding allows solvers to quickly refute the assertion by finding a simple counterexample (e.g., a = 1, b = 1, c = 2).
* This demonstrates the impact of encoding choices on the efficiency of SMT-based verification tools.

### Example 2: Handling Overflow in Integer Addition

However, let's now consider that we have this C program:

````C
#include <assert.h>

int main() {
  int a = nondet_int();
  int b = nondet_int();

  __ESBMC_assume(a > 0 && b > 0);

  assert((a + b) > 0);
}
````

The bit-vector encoding produces this counterexample, which indicates an overflow:

````
[Counterexample]


State 1 file example2.c line 4 column 3 function main thread 0
----------------------------------------------------
  a = 1442565055 (01010101 11111011 11001011 10111111)

State 2 file example2.c line 5 column 3 function main thread 0
----------------------------------------------------
  b = 1103420819 (01000001 11000100 11011101 10010011)

State 4 file example2.c line 9 column 3 function main thread 0
----------------------------------------------------
Violated property:
  file example2.c line 9 column 3 function main
  assertion (a + b) > 0
  a + b > 0


VERIFICATION FAILED
````

While integer encoding proves the program is safe, overflow issues require explicit checks. To prevent overflow, we must add assertion as follows:

````C
#include <assert.h>
#include <limits.h>  // For INT_MAX and INT_MIN

int main() {
  int a = nondet_int();
  int b = nondet_int();

  __ESBMC_assume(a > 0 && b > 0);

  // Check for overflow before asserting the sum is positive
  assert(a <= INT_MAX - b);  // This checks for potential overflow in a + b
  assert((a + b) > 0);  // Check if the sum is greater than 0

  return 0;
}
````

With the added overflow check, the integer encoding now produces a counterexample indicating an overflow:


````
[Counterexample]


State 1 file example3.c line 5 column 3 function main thread 0
----------------------------------------------------
  a = 2147483647

State 2 file example3.c line 6 column 3 function main thread 0
----------------------------------------------------
  b = 1

State 4 file example3.c line 11 column 3 function main thread 0
----------------------------------------------------
Violated property:
  file example3.c line 11 column 3 function main
  assertion a <= INT_MAX - b
  a <= 2147483647 - b


VERIFICATION FAILED
````

We can write an executable test case to confirm this assertion violation:

````
#include <assert.h>
#include <limits.h>  // For INT_MAX and INT_MIN

int main() {
  // Set a and b to values that trigger the assertion failure
  int a = 2147483647;  // INT_MAX, the maximum value for a 32-bit integer
  int b = 1;           // Small positive integer

  // Check for overflow before asserting the sum is positive
  // This assertion will fail since a = INT_MAX and b = 1
  assert(a <= INT_MAX - b);  // Check for potential overflow in a + b

  // This second assertion will not be reached due to the failure above
  assert((a + b) > 0);  // Check if the sum is greater than 0

  return 0;
}
````

````
$ clang example3-test.c -o example3-test
$ ./example3-test
example3-test: example3-test.c:11: int main(): Assertion `a <= INT_MAX - b' failed.
Aborted (core dumped)
````

## Key Observations:

* Bit-vector encoding can detect overflow but may require additional explicit checks to handle cases such as integer overflow.
* The integer encoding efficiently verifies the correctness of the program once overflow checks are introduced, ensuring safe behavior.

## Pseudocode for Counterexample-Guided Abstraction Refinement

The following pseudocode outlines the steps involved in counterexample-guided abstraction refinement:

````
Function VerifyProgram(c_program):
    # Step 1: Add assertions to check for arithmetic overflow
    program_with_overflow_checks = AddOverflowAssertions(c_program)
    
    # Step 2: Invoke ESBMC with the options --z3, --ir, --fixedbv
    esbmc_output = RunESBMC(program_with_overflow_checks, "--z3 --ir --fixedbv")
    
    # Step 3: Check if verification failed
    if VerificationFailed(esbmc_output):
        # Step 4: Extract counterexample from ESBMC output
        counterexample = ExtractCounterexample(esbmc_output)
        
        # Step 5: Create an executable test case from the counterexample
        test_case = CreateTestCase(counterexample)
        
        # Step 6: Compile and run the test case
        result = RunTestCase(test_case)
        
        # Step 7: Check if the assertion is violated
        if AssertionViolated(result):
            # Verification is complete (counterexample is valid)
            Print("Verification complete. Assertion violated.")
        else:
            # Step 8: Add assume statement to the original program to remove spurious counterexample
            updated_program = AddAssumeStatement(c_program, counterexample)
            
            # Go back to Step 2 with the updated program
            VerifyProgram(updated_program)  # Re-run ESBMC with the updated program
    else:
        # Step 9: If verification succeeded, print success message
        Print("Verification succeeded.")
````









