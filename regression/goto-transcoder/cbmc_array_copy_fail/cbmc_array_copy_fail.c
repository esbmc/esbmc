#include <assert.h>

void __CPROVER_array_copy(void *, const void *);

int main(void)
{
  int s[3] = {1, 2, 3};
  int d[3] = {9, 9, 9};
  __CPROVER_array_copy(d, s);
  assert(d[2] == 9);
  return 0;
}
