/* #10504: no call trace is printed when no property is violated. */
#include <assert.h>
int nondet_int(void);
int g;
void setup(int v) { g = v; }
void check(void) { assert(g != 3); }
int main(void)
{
  int v = nondet_int();
  __ESBMC_assume(v != 3);
  setup(v);
  check();
  return 0;
}
