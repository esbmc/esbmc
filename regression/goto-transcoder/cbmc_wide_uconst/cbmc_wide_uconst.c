#include <assert.h>
int main()
{
  unsigned long long x;
  __CPROVER_assume(x == 10000000000ULL);
  assert(x > 0xFFFFFFFFULL); /* 1e10 exceeds 32-bit range */
  return 0;
}
