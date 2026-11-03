/* #8209: the wrapped join of [2, 3] and [3, 0] must keep 2. */
#include <assert.h>
unsigned __VERIFIER_nondet_uint(void);
int __VERIFIER_nondet_int(void);

int main()
{
  unsigned y = __VERIFIER_nondet_uint();
  __ESBMC_assume(y <= 0xFFFFFFFDu);
  unsigned x;
  if (__VERIFIER_nondet_int())
  {
    x = __VERIFIER_nondet_uint();
    __ESBMC_assume(2 <= x && x <= 3);
  }
  else
    x = y + 3u;
  assert(x != 2);
  return 0;
}
