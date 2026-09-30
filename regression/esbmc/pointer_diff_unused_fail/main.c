// An unused pointer difference kept by --no-slice is encoded as a ptrdiff_t.
#include <assert.h>
#include <stddef.h>
unsigned nondet_uint(void);
int main(void)
{
  long a[2];
  ptrdiff_t n = (a + 1) - a;
  (void)n;
  unsigned x = nondet_uint();
  assert(x * 2 != 4);
  return 0;
}
