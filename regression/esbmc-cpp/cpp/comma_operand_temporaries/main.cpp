// A comma's left operand is discarded, but its temporaries live until the end
// of the full-expression ([class.temporary]/4).
#include <cassert>

int dtors = 0;

struct T
{
  int v;
  T(int x) : v(x)
  {
  }
  ~T()
  {
    dtors++;
  }
};

struct U
{
  int seen;
  U() : seen(dtors)
  {
  }
};

int seen()
{
  return dtors;
}

int id(int a)
{
  return a;
}

int ret()
{
  return (T(0), seen());
}

int k = 3;

int main()
{
  int a = (T(1), seen());
  assert(a == 0 && dtors == 1);
  a = ((void)T(2), seen());
  assert(a == 1 && dtors == 2);
  U u = (T(3), (T(4), U()));
  assert(u.seen == 2 && dtors == 4);
  if ((T(5), seen()) != 4)
    assert(0);
  assert(dtors == 5);
  a = id((T(6), seen()));
  assert(a == 5 && dtors == 6);
  a = ret();
  assert(a == 6 && dtors == 7);
  T(7), assert(dtors == 7);
  assert(dtors == 8);
  const int &r = (T(8), k);
  assert(r == 3 && dtors == 9);
  const T &t = (T(9), T(10));
  assert(t.v == 10 && dtors == 10);
  return 0;
}
