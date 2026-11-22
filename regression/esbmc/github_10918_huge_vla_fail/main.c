/* #10918: an object too large to stay on one side of 2^63 must not rule out
 * every path, as a negative VLA bound sign-extended under
 * --no-vla-size-check is on the path that does not declare it (#8048). */
#include <assert.h>
int __VERIFIER_nondet_int(void);
int main()
{
  int n = __VERIFIER_nondet_int();
  if (n > 1)
  {
    int a[n];
  }
  assert(n != -1000);
  return 0;
}
