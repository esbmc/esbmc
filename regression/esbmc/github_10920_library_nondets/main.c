/* esbmc/esbmc#10920: gmtime's operational model assumes ranges on its own
   nondets; they are not program inputs. */
#include <time.h>
int nondet_int();
int main()
{
  time_t t = 0;
  struct tm *p = gmtime(&t);
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 3);
  __ESBMC_assert(x != 2, "x != 2");
  return 0;
}
