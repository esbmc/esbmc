#include <assert.h>
int main()
{
  long long x;
  /* -5000000000 needs 33+ bits; the pre-fix 32-bit constant rewrite truncated
     it and this valid property reported a false FAILED. */
  __CPROVER_assume(x == -5000000000LL);
  assert(x < 0);
  return 0;
}
