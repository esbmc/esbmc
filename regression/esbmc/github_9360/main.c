/* #9360: C17 6.5.3.4p2, a sizeof operand of variable length array type is
   evaluated; any other operand is not. */
#include <assert.h>
#include <stdlib.h>

int c;
int f(void)
{
  return c++;
}

int main(void)
{
  int n = 3, m = 4, i = 0;
  int a[n][m];
  int v[n];

  (void)sizeof(a[i++]);
  assert(i == 1);
  (void)sizeof(a[i++][0]);
  assert(i == 1);

  unsigned s = 0;
  for (int k = 0; k < 2; k++)
    s += sizeof(a[f()]);
  assert(c == 2);
  assert(s == 2 * 4 * sizeof(int));

  int (*p)[m] = &a[0];
  assert(sizeof(*(f(), p)) == 4 * sizeof(int));
  assert(c == 3);
  assert(sizeof(v[f()]) == sizeof(int));
  assert(c == 3);

  int (*q)[n] = malloc(sizeof *q);
  free(q);
  return 0;
}
