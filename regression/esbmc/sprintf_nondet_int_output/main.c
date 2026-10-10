#include <assert.h>
#include <stdio.h>

int nondet_int(void);
unsigned nondet_uint(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x >= -99 && x <= 999);
  char b[8] = "abcdefg";
  int n = sprintf(b, "x%d!", x);
  assert(n >= 3 && n <= 6);
  for (int i = 0; i < 7; i++)
    assert(i < n ? b[i] != '\0' : i > n || b[i] == '\0');
  assert(b[7] == '\0');

  unsigned u = nondet_uint();
  __ESBMC_assume(u < 10);
  char c[4] = "abc";
  int m = snprintf(c, 2, "%u%%%lu", u, 123ul);
  assert(m == 5);
  assert(c[0] != '\0' && c[1] == '\0' && c[2] == 'c');

  char d[3] = "ab";
  int y = nondet_int();
  __ESBMC_assume(y >= -9 && y <= 99);
  sprintf(d, "%i", y);
  assert(d[0] != '\0' && (d[1] == '\0' || d[2] == '\0'));
  return 0;
}
