#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0, last = 0;
  while (i < n)
  {
    i++;
    last = i * 3;
  }
  assert(n == 0 || last == 3 * n);
  return 0;
}
