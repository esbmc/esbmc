/* Basic1_requires_fail: Requires clause violation (design note)
 * Note: This will PASS with --enforce-contract because requires uses ASSUME
 * To detect requires violations, use --replace-call-with-contract instead
 */
#include <assert.h>

int increment(int x)
{


  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value > x);

  return x + 1;
}

int main()
{



  int a = -5;  // VIOLATION: violates requires clause (x > 0)
  int result = increment(a);

  return 0;
}
