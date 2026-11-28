// Virtual base that itself has a base (#11052)
#include <cassert>

struct Root
{
  int r;
  Root() : r(1)
  {
  }
};

struct Base : public Root
{
  int v;
  Base() : v(7)
  {
  }
};

struct Mid : virtual public Base
{
  Mid()
  {
  }
};

int main()
{
  Mid x;
  assert(x.r == 1);
  assert(x.v == 8);
  return 0;
}
