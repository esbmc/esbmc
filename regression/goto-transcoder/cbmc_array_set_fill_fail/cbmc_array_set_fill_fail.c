#include <assert.h>

void __CPROVER_array_set(void *, int);

int main(void)
{
  int a[4];
  __CPROVER_array_set(a, 7);
  assert(a[2] == 8);
  return 0;
}
