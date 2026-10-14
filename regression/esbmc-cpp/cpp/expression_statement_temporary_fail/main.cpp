// The temporary passed to get() is destroyed when the statement ends, so
// nothing is live at the assertion.
#include <cassert>

int live = 0;

struct C
{
  int v;
  C(int x) : v(x)
  {
    ++live;
  }
  ~C()
  {
    --live;
  }
};

int get(const C &c)
{
  return c.v;
}

int main()
{
  get(C(1));
  assert(live == 1);
  return 0;
}
