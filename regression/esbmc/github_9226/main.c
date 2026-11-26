/* github #9226: byte offsets of elements and members, including a negative
 * index, and of a variable-length row, which keeps the bit-level offset. */
#include <assert.h>

int nondet_int(void);
int a[10];
short m[3][5];
struct s
{
  char c;
  int x[3];
} st[4];

int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 3);
  assert((char *)&a[i] - (char *)a == 4 * i);
  assert((char *)&m[i][i + 1] - (char *)m == 10 * i + 2 * (i + 1));
  assert((char *)&st[i].x[i] - (char *)st == 16 * i + 4 + 4 * i);

  int *p = &a[5];
  int j = -i;
  assert(&p[j] == a + 5 - i);

  int n = nondet_int();
  __ESBMC_assume(n > 0 && n < 4 && i < n);
  int v[n][n];
  assert((char *)&v[i][i] - (char *)v == 4 * (n * i + i));
  return 0;
}
