/* #9000: a pointer that may be &s or a heap object too small for struct P
 * does not satisfy is_fresh. */
#include <stdlib.h>
_Bool nondet_bool();
struct P
{
  int a;
  int b;
};
void g(struct P *d)
{
  __ESBMC_requires(__ESBMC_is_fresh(d, sizeof(struct P)));
  __ESBMC_ensures(1);
  d->b = 1;
}
int main()
{
  struct P s;
  struct P *d = nondet_bool() ? &s : malloc(sizeof(int));
  g(d);
  return 0;
}
