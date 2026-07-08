#include <assert.h>
int main()
{
  float a, b;
  __CPROVER_assume(a == 1.5f);
  __CPROVER_assume(b == 2.5f);
  assert(a != b);
  return 0;
}
