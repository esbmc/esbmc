#include <assert.h>

struct S
{
  int f;
};

// A store through `p->f` writes to `a.f`, so `a.f + b` must be recomputed.
// The dereference is nested inside the lvalue rather than being it, which
// GCSE used to miss.
int main()
{
  struct S a;
  a.f = 1;
  int b = 2;
  struct S *p = &a;

  int x = a.f + b;
  p->f = 100;
  int y = a.f + b;

  assert(y == 102);
  return 0;
}
