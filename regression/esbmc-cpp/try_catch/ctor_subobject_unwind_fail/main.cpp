// The member built before the throwing one is destroyed ([except.ctor]/3), so
// the assertion below fails natively.
#include <cassert>

int dtors = 0;

struct C
{
  C(int x)
  {
    if (x == 0)
      throw 1;
  }
  ~C()
  {
    ++dtors;
  }
};

struct P
{
  C a;
  C b;
  P() : a(1), b(0)
  {
  }
};

int main()
{
  try
  {
    P p;
  }
  catch (int)
  {
  }
  assert(dtors == 0);
}
