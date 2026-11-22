/* esbmc/esbmc#10920: k[0] is also read inside the isdigit model, which does
   not make it a library input. */
#include <ctype.h>
int nondet_int();
int main()
{
  int k[1];
  k[0] = nondet_int();
  __ESBMC_assume(k[0] >= 0 && k[0] < 4);
  int r = isdigit(k[0]);
  __ESBMC_assert(k[0] < 2 || r, "k[0] < 2 || r");
  return 0;
}
