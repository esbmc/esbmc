#include <assert.h>

/* #8240: __CPROVER_OBJECT_SIZE and __CPROVER_r_ok count bytes, not elements. */
struct S
{
  int x;
  int y;
};

int main(void)
{
  int a[4];
  double d[3];
  struct S s;
  assert(__CPROVER_OBJECT_SIZE(a) == 16);
  assert(__CPROVER_OBJECT_SIZE(d + 1) == 24);
  assert(__CPROVER_OBJECT_SIZE(&s.y) == 8);
  assert(__CPROVER_r_ok(a, 16));
  assert(!__CPROVER_r_ok(a, 17));
  assert(__CPROVER_r_ok(a + 1, 12));
  assert(!__CPROVER_r_ok(&s.y, 5));
  return 0;
}
