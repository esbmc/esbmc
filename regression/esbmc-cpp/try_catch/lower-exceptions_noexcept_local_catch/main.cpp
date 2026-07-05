#include <cassert>

// A throw that is caught within the noexcept function does not escape, so it
// does not violate the specification.
struct E
{
  int v;
  E(int a) : v(a)
  {
  }
};

void f() noexcept
{
  try
  {
    throw E(5);
  }
  catch (E &e)
  {
    assert(e.v == 5);
  }
}

int main()
{
  f();
  return 0;
}
