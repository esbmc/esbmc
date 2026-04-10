/*
 * Test: loop-invariant + replace-call-with-contract combination
 *
 * Verifies that k-induction loop invariants work correctly when
 * function calls inside the loop are replaced with their contracts.
 *
 * step(x): contracted function returning x+1
 * main(): loop calling step() with invariant tracking i and sum
 *
 * Command:
 *   esbmc main.c --loop-invariant --replace-call-with-contract "step"
 */

#include <assert.h>

int step(int x) {
    __ESBMC_requires(x >= 0 && x < 100);
    __ESBMC_ensures(__ESBMC_return_value == x + 1);

    return x + 1;
}

int main() {
    int i = 0;
    int sum = 0;

    __ESBMC_loop_invariant(i >= 0 && i <= 10);
    __ESBMC_loop_invariant(sum == i);
    while (i < 10) {
        i = step(i);
        sum++;
    }

    assert(i == 10);
    assert(sum == 10);
    return 0;
}
