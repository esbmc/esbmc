// Issue #9185: the DEAD of p ran before the throw read *p.
#include <cassert>

int g = 7;

void f()
{
  int *p = &g;
  throw *p;
}

int main()
{
  try
  {
    f();
  }
  catch (int v)
  {
    assert(v == 7);
    assert(v == 8);
  }
}
