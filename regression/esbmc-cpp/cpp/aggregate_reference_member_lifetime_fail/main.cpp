// A temporary bound to a reference member of a braced aggregate initialiser
// lives as long as the aggregate ([class.temporary]/6), so it is not yet
// destroyed: each assertion below fails natively.
#include <cassert>

int live = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++live;
  }
  ~M()
  {
    --live;
  }
};

struct R
{
  const M &m;
};

struct O
{
  R r;
};

struct RR
{
  M &&m;
};

int main()
{
  {
    R x{M(1)};
    assert(live == 0);
  }
  {
    O o{{M(2)}};
    assert(live == 0);
  }
  {
    RR r = {M(3)};
    assert(live == 0);
  }
  return 0;
}
