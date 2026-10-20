#include <assert.h>

/* An array too large to place above the first page must not make its path
 * infeasible: the assertion stays reachable. */
int main(void)
{
  unsigned long n = nondet_ulong();
  if (n > 18446744073709551615ul - 4000 && n <= 18446744073709551615ul - 16)
  {
    char a[n];
    a[0] = 1;
    assert((unsigned long)a == 0);
  }
  return 0;
}
