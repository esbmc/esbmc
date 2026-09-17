#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned arr[64] = {0};
  unsigned (*p)[64] = &arr;
  unsigned i = 0;
  while (i < n)
  {
    i++;
    (*p)[0] = (*p)[0] + 1;
  }
  assert(arr[0] == 0);
  return 0;
}
