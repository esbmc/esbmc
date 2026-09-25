/* __builtin_isinf_sign lowers to isinf ? (signbit ? -1 : 1) : 0. It is spelled
   exactly, not by the base name "isinf" a program may reuse, and left as a call
   the symbol is bodyless -- so the result is nondet and none of these holds. */
#include <assert.h>
#include <math.h>

int main(void)
{
  assert(__builtin_isinf_sign(1.0) == 0);
  assert(__builtin_isinf_sign(INFINITY) == 1);
  assert(__builtin_isinf_sign(-INFINITY) == -1);
  return 0;
}
