#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0, j = 0;
  while (i < n)
  {
    if (i >= 10)
      break;
    i++;
    j += 2;
  }
  assert(i <= 9);
  return 0;
}
