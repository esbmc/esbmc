// Test: Extern variable without __ESBMC_EXTERN_NOVAL gets nondet value
//
// This simulates user code that includes an operational model header.
// Without the __ESBMC_EXTERN_NOVAL attribute on the extern declaration,
// ESBMC assigns a nondeterministic value to the variable.
//
// The assertion fails because 'counter' can be any value (the SMT solver
// finds a counterexample), rather than the value defined in the library.
//
// Compare with the esbmc_noval test where __ESBMC_EXTERN_NOVAL prevents
// the nondet assignment, allowing the library's definition to be used.

#include "header.h"

int main() {
    __ESBMC_assert(counter == 42, "Fails: counter is nondet, not 42");
    return 0;
}
