/* Minimal test case for multiple __ESBMC_old in single expression causing crash
 * 
 * Bug: When --enforce-contract is used with ensures containing multiple
 * __ESBMC_old() calls in a single expression (especially in conditional),
 * ESBMC crashes with:
 *   Assertion `t->target_number != 0' failed
 *   in goto_program.cpp:378
 * 
 * This reproduces the exact pattern from Fibonacci.c::fib_reaction_timer
 * where multiple __ESBMC_old appear in the right side of || operator.
 */
#include <stddef.h>

typedef struct {
    int N;
    int lastResult;
    int secondLastResult;
    int result;
} Fib;

void fib_reaction_timer(Fib *self) {
    __ESBMC_requires(self != NULL);
    
    __ESBMC_assigns(self->result);
    // Pattern from Fibonacci.c - multiple __ESBMC_old in conditional
    __ESBMC_ensures(__ESBMC_old(self->N) >= 2 || self->result == 1);
    __ESBMC_ensures(__ESBMC_old(self->N) < 2 || 
                    self->result == __ESBMC_old(self->lastResult) + __ESBMC_old(self->secondLastResult));
    
    if (self->N < 2) {
        self->result = 1;
    } else {
        self->result = self->lastResult + self->secondLastResult;
    }
}

int main() {
    Fib fib = {5, 3, 2, 0};
    fib_reaction_timer(&fib);
    return 0;
}
