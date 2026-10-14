#include <cassert>

// noexcept(false) explicitly permits exceptions to escape, so a throw is NOT a
// specification violation: it propagates normally and is caught by the caller.
struct E
{
  int v;
  E(int a) : v(a)
  {
  }
};

void f() noexcept(false)
{
  throw E(5);
}

int main()
{
  try
  {
    f();
  }
  catch (E &e)
  {
    assert(e.v == 5);
    return 1;
  }
  return 0;
}
