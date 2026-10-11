#include <cassert>

int ctors = 0, dtors = 0;

struct C
{
  int v;
  C(int x) : v(x)
  {
    ctors++;
  }
  C(const C &o) : v(o.v)
  {
    ctors++;
  }
  ~C()
  {
    dtors++;
  }
};

int main()
{
  try
  {
    throw C(1);
  }
  catch (C c)
  {
  }
  assert(ctors == dtors);
  return 0;
}
