#include <assert.h>

int main()
{
  char a[4];
  assert(__CPROVER_rw_ok(a, 8));
  return 0;
}
