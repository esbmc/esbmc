#include <assert.h>
int main()
{
  int x = 5;
  int *p = &x;
  __CPROVER_assert(__CPROVER_POINTER_OFFSET(p) == 0, "zero offset");
  return 0;
}
