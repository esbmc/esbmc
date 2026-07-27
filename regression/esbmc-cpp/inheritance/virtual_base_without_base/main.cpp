// Control for virtual_base_with_base: the same shape, but the virtual base has
// no base of its own. This one works, which is what isolates the trigger.
#include <cassert>

struct Base
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
  assert(x.v == 7);
  return 0;
}
