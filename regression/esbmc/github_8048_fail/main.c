/* #8048: a VLA declared on a path not taken must not rule out the paths where
 * its bound is negative. */
#include <assert.h>
int __VERIFIER_nondet_int(void);
int main()
{
  int n = __VERIFIER_nondet_int();
  if (n > 1)
  {
    int a[n];
  }
  assert(n != -1);
  return 0;
}
