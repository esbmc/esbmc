// A temporary that initialises an aggregate's element is that element
// ([dcl.init.aggr]/4): only the aggregate's destructor destroys it.
#include <cassert>

int ctors = 0, dtors = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++ctors;
  }
  M(const M &o) : v(o.v)
  {
    ++ctors;
  }
  ~M()
  {
    ++dtors;
  }
};

struct W
{
  int k;
  M a, b;
};

M make(int x)
{
  return M(x);
}

int main()
{
  {
    W w{1, M(2), make(3)};
    M arr[2] = {M(4), make(5)};
    assert(w.b.v == 3 && arr[1].v == 5);
    assert(dtors == 0);
  }
  assert(ctors == 4 && dtors == 4);
  return 0;
}
