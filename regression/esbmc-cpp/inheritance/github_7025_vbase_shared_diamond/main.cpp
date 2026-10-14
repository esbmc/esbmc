// esbmc/esbmc#7025 residual: a virtual base shared by two sibling bases has no
// single displacement in the flattened layout, so the upcast to R is left
// unadjusted and R::r is written at the wrong offset.
#include <cassert>

struct V
{
  int v;
};
struct L : virtual V
{
  int l;
};
struct R : virtual V
{
  int r;
};
struct D : L, R
{
};

int main()
{
  D d;
  R *pr = &d;
  pr->r = 7;
  assert(d.r == 7);
  return 0;
}
