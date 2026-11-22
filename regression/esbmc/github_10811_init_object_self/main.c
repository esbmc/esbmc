#include <assert.h>
int x = 0;
void *p;
int main()
{
  p = nondet_bool() ? (void *)&x : (void *)&p;
  __ESBMC_assume(p == (void *)&p);
  __ESBMC_init_object(p);
  assert(x == 0);
  return 0;
}
