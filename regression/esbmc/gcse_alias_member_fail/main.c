#include <assert.h>

struct S
{
  int f;
};

// Negative counterpart of gcse_alias_member: 3 is the stale value `a.f + b`
// held before the store through `p->f`. If GCSE ever stops invalidating it,
// this assertion starts holding and the test fails.
int main()
{
  struct S a;
  a.f = 1;
  int b = 2;
  struct S *p = &a;

  int x = a.f + b;
  p->f = 100;
  int y = a.f + b;

  assert(y == 3);
  return 0;
}
