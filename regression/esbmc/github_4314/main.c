#include <assert.h>
extern int nondet_int(void);

/* The assertion is violable at several unwindings, so an incremental run
   enumerates witnesses at more than one k. Without the bound in the header
   the blocks are indistinguishable from a repeat of the same one. */
int main()
{
  int n = nondet_int();
  __ESBMC_assume(n >= 0 && n <= 12);
  int s = 0;
  for (int i = 0; i < n; i++)
    s += 1;
  assert(s != 2 && s != 3);
  return 0;
}
