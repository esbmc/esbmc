#include <assert.h>
int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0;
  while (i < n)
  {
    ++i;
    assert(i <= n);
  }
  return 0;
}
