#include <stdlib.h>
#include <assert.h>
int main()
{
  long long x;
  __CPROVER_assume(x == -2000000000LL);
  assert(llabs(x) == 2000000000LL);
  return 0;
}
