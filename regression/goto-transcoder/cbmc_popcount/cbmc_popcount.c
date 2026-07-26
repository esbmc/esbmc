#include <assert.h>
int main()
{
  unsigned x;
  __CPROVER_assume(x == 0xFFu);
  assert(__builtin_popcount(x) == 8);
  return 0;
}
