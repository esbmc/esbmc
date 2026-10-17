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

struct W
{
  M m;
};

struct V
{
  W w;
  V() : w{M(1)}
  {
  }
};

int main()
{
  V v;
  assert(live == 0);
  return 0;
}
