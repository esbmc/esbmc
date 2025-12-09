#include <assert.h>
typedef struct
{
  _Bool (*a)();
} c;
void *e;
c f[1];
c *g = f;
_Bool b()
{
  return 1;
}
void d();
void main()
{
  int i = nondet_int();
  f[0] = (c){b};
  __ESBMC_assume(i >= 0 && i < 1);
  if (g[i].a())
    e = d;
  assert(e);
}
