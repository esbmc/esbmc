#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0, last = 0, prev = 0;
  while (i < n)
  {
    i++;
    last = prev;
    prev = i;
  }
  assert(i == 0 || last + 2 == prev);
  return 0;
}
