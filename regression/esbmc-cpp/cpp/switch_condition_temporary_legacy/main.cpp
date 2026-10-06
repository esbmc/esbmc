// A temporary in a switch condition is destroyed at the end of the condition,
// before control reaches a case label ([class.temporary]/4).
#include <cassert>

int nondet_int();

int dtors = 0;

struct T
{
  int v;
  T(int x) : v(x)
  {
  }
  ~T()
  {
    ++dtors;
  }
};

int pick(int n)
{
  switch (T(n).v)
  {
  case 1:
    return dtors;
  case 2:
    break;
  default:
    return -dtors;
  }
  return 10 + dtors;
}

int main()
{
  int x = nondet_int();
  switch (T(x).v)
  {
  case 4:
    assert(dtors == 1);
    switch (T(x + 1).v)
    {
    case 5:
      assert(dtors == 2);
    }
    break;
  default:
    assert(dtors == 1);
  }
  assert(dtors == (x == 4 ? 2 : 1));

  dtors = 0;
  assert(pick(1) == 1);
  assert(pick(2) == 12);
  assert(pick(3) == -3);
  assert(dtors == 3);
}
