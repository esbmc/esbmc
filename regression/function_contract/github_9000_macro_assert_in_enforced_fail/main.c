/* #9000: a user assertion that shares a macro expansion with a clause of an
 * enforced function stays in the function's body. */
#include <assert.h>
#define SPEC(x)                                                                \
  assert((x) > 5);                                                             \
  __ESBMC_requires((x) > 0)
int f(int a)
{
  SPEC(a);
  __ESBMC_ensures(__ESBMC_return_value > 0);
  return a;
}
int main()
{
  return 0;
}
