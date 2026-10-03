// A temporary that initialises an aggregate's element is that element
// ([dcl.init.aggr]/4), so it is not destroyed at the end of the
// full-expression: each assertion below fails natively.
#include <cassert>

int dtors = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
  }
  ~M()
  {
    ++dtors;
  }
};

struct W
{
  M m;
};

M make(int x)
{
  return M(x);
}

int main()
{
  {
    W w{M(1)};
    assert(dtors == 1);
  }
  int before = dtors;
  {
    M arr[1] = {M(2)};
    assert(dtors == before + 1);
  }
  before = dtors;
  {
    W w{make(3)};
    assert(dtors == before + 1);
  }
  return 0;
}
