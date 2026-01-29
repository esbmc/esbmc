// Test case: Verify that precondition violations are detected in REPLACE phase
// The caller violates the function's precondition
// Expected: Verification should fail with "contract requires (replace)"

#include <assert.h>

// Function with pragma contract and an explicit precondition
// Requires that the pointer is valid and value is positive
#pragma contract
void increment_positive(int *x)
{
  __ESBMC_requires(x != 0);
  __ESBMC_requires(*x > 0);
  __ESBMC_ensures(*x == __ESBMC_old(*x) + 1);
  __ESBMC_assigns(*x);
  
  *x = *x + 1;
}

int main()
{
  int value = -5;  // BUG: negative value violates precondition (*x > 0)
  increment_positive(&value);  // This call should fail in REPLACE phase
  
  return 0;
}
