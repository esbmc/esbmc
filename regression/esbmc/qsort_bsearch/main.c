#include <assert.h>
#include <stdlib.h>

int nondet_int(void);

struct pair
{
  int key;
  double value;
};

static int cmp_int(const void *a, const void *b)
{
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}

static int cmp_double(const void *a, const void *b)
{
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static int cmp_pair(const void *a, const void *b)
{
  return cmp_int(
    &((const struct pair *)a)->key, &((const struct pair *)b)->key);
}

int main(void)
{
  int a[4] = {3, 1, 4, 2};
  qsort(a, 4, sizeof a[0], cmp_int);
  assert(a[0] == 1 && a[1] == 2 && a[2] == 3 && a[3] == 4);

  int k = 3;
  assert(bsearch(&k, a, 4, sizeof a[0], cmp_int) == &a[2]);
  k = 5;
  assert(bsearch(&k, a, 4, sizeof a[0], cmp_int) == NULL);

  int x = nondet_int(), y = nondet_int(), z = nondet_int();
  int b[3] = {x, y, z};
  qsort(b, 3, sizeof b[0], cmp_int);
  assert(b[0] <= b[1] && b[1] <= b[2]);
  assert(b[0] == x || b[0] == y || b[0] == z);
  assert(bsearch(&y, b, 3, sizeof b[0], cmp_int) != NULL);

  double d[3] = {2.5, -1.0, 1e300};
  qsort(d, 3, sizeof d[0], cmp_double);
  assert(d[0] == -1.0 && d[1] == 2.5 && d[2] == 1e300);

  struct pair p[3] = {{2, 0.5}, {0, 1.5}, {1, 2.5}};
  qsort(p, 3, sizeof p[0], cmp_pair);
  assert(p[0].key == 0 && p[0].value == 1.5);
  assert(p[2].key == 2 && p[2].value == 0.5);
  return 0;
}
