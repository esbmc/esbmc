/* address_of2t's primary constructor takes the *pointee* type, so re-attaching
   a pointer argument's own type through expr2t::with_type widened `int *` to
   `int **` and the store reported a spurious out-of-bounds. */
#include <assert.h>

int main(void)
{
  int x = 0x7fffffff;
  int y;
  _Bool c = __builtin_sadd_overflow(2, x, &y);
  assert(c);

  unsigned int ux = 2;
  unsigned int uy;
  _Bool uc = __builtin_uadd_overflow(2, ux, &uy);
  assert(!uc);
  assert(uy == 4);
  return 0;
}
