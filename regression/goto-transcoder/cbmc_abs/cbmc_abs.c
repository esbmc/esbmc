#include <stdlib.h>
#include <assert.h>
int main()
{
  int x;
  __CPROVER_assume(x == -7);
  assert(abs(x) == 7);
  return 0;
}
