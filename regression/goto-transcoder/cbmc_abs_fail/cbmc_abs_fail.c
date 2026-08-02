#include <stdlib.h>
#include <assert.h>
int main()
{
  int x;
  __CPROVER_assume(x == -7);
  assert(abs(x) == -7); /* wrong: abs(-7) == 7, not -7 */
  return 0;
}
