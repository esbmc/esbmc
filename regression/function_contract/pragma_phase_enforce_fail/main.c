// Test case: Verify that postcondition violations are detected in ENFORCE phase
// This function violates its postcondition (ensure clause)
// Expected: Verification should fail with "contract ensures (enforce)"

#include <assert.h>

// Function with pragma contract and an explicit postcondition
// The implementation violates the postcondition
#pragma contract
void increment_by_two(int *x)
{
  __ESBMC_requires(x != 0);
  __ESBMC_ensures(*x == __ESBMC_old(*x) + 2);
  __ESBMC_assigns(*x);
  
  // BUG: Only increments by 1, but postcondition requires increment by 2
  *x = *x + 1;
}

int main()
{
  int value = 10;
  // This should fail during contract enforcement (enforce phase)
  // because the implementation violates the postcondition
  increment_by_two(&value);
  return 0;
}
