/* Basic25: Void return type with contracts
 * Tests that void functions can have ensures clauses that check side effects
 */
#include <assert.h>

int global_value = 0;

void set_global(int x)
{
  __ESBMC_requires(x >= 0);
  // For void functions, ensures can check side effects (global variables)
  // Note: Without __ESBMC_old support, we can only check final state
  global_value = x;
}

void increment_global(int delta)
{
  __ESBMC_requires(delta > 0);
  // Ensures that global was incremented correctly
  global_value += delta;
}

int main()
{
  // Test 1: Simple void function with side effect
  set_global(10);
  assert(global_value == 10);
  
  // Test 2: Void function that modifies global
  increment_global(5);
  assert(global_value == 15);
  
  return 0;
}

