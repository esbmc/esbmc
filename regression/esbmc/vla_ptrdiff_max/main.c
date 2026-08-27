/* The boundary of R40's report: exactly PTRDIFF_MAX is the largest VLA whose
   one-past-the-end offset still reads non-negative in the pointer comparator,
   so it is declared rather than reported. */
#include <assert.h>
#include <stdint.h>

int main(void)
{
  uint64_t n = 0x7FFFFFFFFFFFFFFFUL; /* PTRDIFF_MAX */

  char a[n];
  char *q = a + n;

  assert(q >= a);
  return 0;
}
