#include <assert.h>
int nondet_int();
int x[2], y[2];
int main()
{
  x[0] = 1;
  x[1] = 2;
  y[0] = 3;
  y[1] = 4;
  int z = nondet_int(); // input
  int *r1 = &x[1];
  int *r2 = &y[1];
  __ESBMC_loop_invariant((r1 == &x[1] && r2 == &y[1]) || (r1 == &y[1] && r2 == &x[1]) && x[1] == 2 && y[1] == 4);
  while(z > 0)
  {
    int *tmp = r1;
    r1 = r2;
    r2 = tmp;
    z = z - 1;
  }
  assert(*r1 == 2 || *r1 == 4);
}
