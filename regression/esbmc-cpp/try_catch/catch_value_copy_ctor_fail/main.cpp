#include <cassert>

int copies = 0;

struct C
{
  int v;
  C(int x) : v(x)
  {
  }
  C(const C &o) : v(o.v)
  {
    copies++;
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
    assert(c.v == 1);
    assert(copies == 0);
  }
  return 0;
}
