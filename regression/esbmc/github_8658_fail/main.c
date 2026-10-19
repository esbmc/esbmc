/* Issue #8658: INT_MIN + INT_MIN wraps to 0 and was not reported. */
#include <limits.h>
int nondet_int();
int main(void)
{
  int a = nondet_int();
  __ESBMC_assume(a == INT_MIN);
  int r = a + a;
  return r;
}
