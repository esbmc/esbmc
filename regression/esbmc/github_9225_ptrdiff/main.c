/* github #9225: pointer differences divide by a power-of-two element size
 * with a shift, truncating toward zero as the division does. */
#include <assert.h>

unsigned nondet_uint(void);
int a[10];
double d[10];
char c[10];
struct s
{
  int a, b, c;
} st[10];

int main(void)
{
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 10);
  int *p = &a[i];
  assert(p - a == i);
  assert(a - p == -(long)i);
  assert(&d[i] - d == i);
  assert(d - &d[i] == -(long)i);
  assert(&c[i] - c == i);
  assert(c - &c[i] == -(long)i);
  assert(&st[i] - st == i);
  assert(st - &st[i] == -(long)i);

  /* 6 bytes is not a whole number of ints: +6/4 and -6/4 truncate to 1, -1. */
  int *q = (int *)((char *)a + 6);
  assert(q - a == 1);
  assert(a - q == -1);
  return 0;
}
