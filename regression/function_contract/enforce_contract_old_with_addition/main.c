/* Minimal test case for ensures with __ESBMC_old addition expression
 * 
 * Bug: When --enforce-contract is used with ensures containing __ESBMC_old()
 * in an addition expression like: result == __ESBMC_old(a) + __ESBMC_old(b)
 * ESBMC crashes because the GOTO target (labeled ensures ASSUME) is removed
 * but GOTO instructions still point to it.
 *
 * Pattern from Fibonacci.c::fib_reaction_timer:
 *   __ESBMC_ensures(__ESBMC_old(self->N) < 2 || 
 *                   self->result == __ESBMC_old(self->lastResult) + __ESBMC_old(self->secondLastResult));
 */
#include <stddef.h>

typedef struct {
    int N;
    int result;
    int lastResult;
    int secondLastResult;
} State;

// Mimics Fibonacci.c::fib_reaction_timer pattern with two pointer parameters
void compute(State *self, int *out)
{
    __ESBMC_requires(self != NULL);
    __ESBMC_requires(out != NULL);
    
    __ESBMC_assigns(self->result, *out);
    
    // Case 1: if N < 2, then result == 1
    __ESBMC_ensures(__ESBMC_old(self->N) >= 2 || self->result == 1);
    // Case 2: if N >= 2, then result == lastResult + secondLastResult
    __ESBMC_ensures(__ESBMC_old(self->N) < 2 || self->result == __ESBMC_old(self->lastResult) + __ESBMC_old(self->secondLastResult));
    __ESBMC_ensures(*out == self->result);
    
    if (self->N < 2) {
        self->result = 1;
    } else {
        self->result = self->lastResult + self->secondLastResult;
    }
    *out = self->result;
}

int main()
{
    State state = {5, 0, 3, 2};
    int out;
    compute(&state, &out);
    return 0;
}
