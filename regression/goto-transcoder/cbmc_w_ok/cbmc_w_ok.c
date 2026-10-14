#include <assert.h>
int main()
{
  int x = 5;
  int *p = &x;
  __CPROVER_assert(__CPROVER_w_ok(p, sizeof(int)), "writable");
  *p = 10;
  assert(x == 10);
  return 0;
}
