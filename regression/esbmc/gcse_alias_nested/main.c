#include <assert.h>

struct S
{
  int f;
};

// `(*q)->f` is a dereference of a dereference, which the points-to analysis
// never interned as a node. Reporting an empty reference set for it read as
// "refers to no object", letting GCSE keep `a.f + b` across the store.
int main()
{
  struct S a;
  a.f = 1;
  int b = 2;
  struct S *p = &a;
  struct S **q = &p;

  int x = a.f + b;
  (*q)->f = 100;
  int y = a.f + b;

  assert(y == 102);
  return 0;
}
