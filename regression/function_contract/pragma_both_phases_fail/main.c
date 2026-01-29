/* Test: Both enforce-phase and replace-phase failures
 * 
 * This test demonstrates that both phases can fail independently:
 * - Enforce phase: Function implementation violates postcondition
 * - Replace phase: Caller violates precondition
 * 
 * With --parallel-solving, both failures will be detected simultaneously
 */
#include <assert.h>

#pragma contract
void increment(int *x)
{
  __ESBMC_requires(x != 0);
  __ESBMC_requires(*x > 0);
  __ESBMC_ensures(*x == __ESBMC_old(*x) + 1);
  __ESBMC_assigns(*x);
  
  // BUG in implementation: increments by 2 instead of 1
  *x = *x + 2;  // Violates ensures (enforce-phase will fail)
}

int main(void)
{
  int value = -5;  // BUG: negative value violates requires (replace-phase will fail)
  
  increment(&value);
  
  return 0;
}
