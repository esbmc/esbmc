// Negative counterpart of reference_temporary_single_dtor: the reference binds
// directly to a single materialised temporary, so exactly one destructor runs
// and g is 1 (not 2, the old double-construct/destroy count). Asserting g == 2
// must fail.
#include <cassert>

int g;

struct B
{
  int x;
  B(int v) : x(v)
  {
  }
  ~B()
  {
    g++;
  }
};

int main()
{
  {
    const B &r = B(7);
    assert(r.x == 7);
  }
  assert(g == 2);
  return 0;
}
