#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0;
  while (i < n)
  {
    i++;
    assert(i > 1);
  }
  return 0;
}
