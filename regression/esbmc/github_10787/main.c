// Addresses up to one past the end are valid (C11 6.5.6p8): a member array, a
// row of a two-dimensional array (written by the frontend as &m[i][0]), a VLA,
// and a trailing array reached through a pointer to a larger allocation.
#include <assert.h>
#include <stdlib.h>
struct
{
  int a[1];
  int end;
} b;
int m[2][3];
struct hack
{
  int n;
  int d[1];
};
int main()
{
  int c = 1;
  int *p = &b.a[c];
  int i = 2;
  int (*row)[3] = &m[i];
  int k = nondet_int();
  __ESBMC_assume(k >= 0 && k <= 3);
  int *q = &m[1][k];
  int n = nondet_int();
  __ESBMC_assume(n >= 1 && n <= 4);
  int v[n];
  int *w = &v[n];
  struct hack *h = malloc(sizeof(struct hack) + 4 * sizeof(int));
  if (h)
  {
    int *e = &h->d[4];
    assert(e == h->d + 4);
  }
  assert(p == b.a + 1 && row == m + 2 && q == m[1] + k && w == v + n);
  return 0;
}
