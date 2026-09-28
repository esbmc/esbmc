// Non-vacuity guard for valarray_members: max() really computes, so asserting
// the wrong maximum must FAIL. Before the fix this failed too -- but so did the
// positive form, which is the tell-tale of a nondet return.
#include <valarray>
#include <cassert>

int main()
{
  std::valarray<int> a(3);
  a[0] = 5;
  a[1] = 2;
  a[2] = 9;
  assert(a.max() == 5);
  return 0;
}
