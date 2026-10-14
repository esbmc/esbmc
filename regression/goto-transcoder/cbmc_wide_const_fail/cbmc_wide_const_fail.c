#include <assert.h>
int main()
{
  long long x;
  __CPROVER_assume(x == 5000000000LL);
  assert(x < 4000000000LL); /* wrong: 5e9 is not < 4e9 */
  return 0;
}
