#include <cassert>

struct Base
{
  int x;
  Base(int x) : x(x)
  {
  }
};
struct Derived : Base
{
  using Base::Base;
};

int main()
{
  Derived d{2};
  assert(d.x == 2);
}
