#include <cassert>

// catch (T*): the thrown pointer is stored and read back (p = *(T**)value),
// so the handler sees the original address and its pointee.
struct A
{
  int x;
};

int main()
{
  A a;
  a.x = 5;
  try
  {
    throw &a;
  }
  catch (A *p)
  {
    assert(p->x == 5);
    return 1;
  }
  return 0;
}
