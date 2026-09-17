#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0, s = 0;
  while (i < n)
  {
    i++;
    s++;
  }
  assert(s == n + 1);
  return 0;
}
