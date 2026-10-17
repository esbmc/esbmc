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

int main()
{
  W *w = new W{M(1)};
  *w = W{M(5)};
  assert(live == 0);
  delete w;
  return 0;
}
