/* #9000: a user assertion that shares a macro expansion, and so a location,
 * with a clause survives when the clauses of an inactive function are
 * dropped. */
#include <assert.h>
#define SPEC(x)                                                                \
  assert((x) > 5);                                                             \
  __ESBMC_requires((x) > 0)
int g(int x)
{
  SPEC(x);
  return x;
}
int f(int a)
{
  __ESBMC_requires(a > 0);
  __ESBMC_ensures(__ESBMC_return_value > 0);
  return a;
}
int main()
{
  int v = 1;
  g(v);
  return 0;
}
