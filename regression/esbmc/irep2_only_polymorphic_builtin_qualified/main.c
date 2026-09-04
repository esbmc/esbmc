#include <assert.h>

/* The IREP2 seam has no representation for C type qualifiers, so a volatile
 * and a plain call to the same builtin collapse onto one instance where
 * clang_c_adjust makes two (_vS32 and _S32). The instance is shared, but the
 * arithmetic each call site performs must still be its own. */
volatile int vi;
int pi;

int main(void)
{
  vi = 1;
  pi = 10;

  assert(__sync_fetch_and_add(&vi, 2) == 1);
  assert(__sync_fetch_and_add(&pi, 5) == 10);
  assert(vi == 3);
  assert(pi == 15);

  return 0;
}
