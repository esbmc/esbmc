#include <assert.h>
int main()
{
  unsigned x;
  __CPROVER_assume(x == 1u);
  assert(__builtin_clz(x) == 30);
  return 0;
}