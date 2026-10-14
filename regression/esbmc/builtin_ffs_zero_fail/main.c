// ffs(0) is 0, not the width-plus-one its ctz-based encoding would give if the
// zero case were dropped. Guards the special case against being folded away.
// --clz-zero-check must also leave ffs alone: zero is defined here, so the
// undefined-behaviour claim the flag adds for clz/ctz does not apply. #183
#include <assert.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x == 0);
  assert(__builtin_ffs(x) == 1);
  return 0;
}
