#include <assert.h>
int main()
{
  float a, b;
  __CPROVER_assume(a == 3.0f);
  __CPROVER_assume(b == 3.0f);
  assert(a != b);
  return 0;
}
