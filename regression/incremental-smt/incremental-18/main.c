
#include <assert.h>
int nondet_int();
int main()
{
  int x = nondet_int(); // input
  int y = nondet_int(); // input
  int z = nondet_int(); // input
  int *r1 = &x;
  int *r2 = &y;
  __ESBMC_loop_invariant((r1 == &x && r2 == &y) || (r1 == &y && r2 == &x));
  while(z > 0)
  {
    int *tmp = r1;
    r1 = r2;
    r2 = tmp;
    z = z - 1;
  }
  assert(*r1 == x || *r1 == y);
}
