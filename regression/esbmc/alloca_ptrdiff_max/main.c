/* The boundary of R39's report: exactly PTRDIFF_MAX is the largest request
   whose one-past-the-end offset still reads non-negative in the pointer
   comparator, so it is allocated rather than reported. */
#include <assert.h>
#include <stdint.h>

int main(void)
{
  const uint64_t n = 0x7FFFFFFFFFFFFFFFUL; /* PTRDIFF_MAX */

  char *p = __builtin_alloca(n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
