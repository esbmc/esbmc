#include <cassert>

struct Base
{
  int a;
  int b;
  Base(int x, int y) : a(x), b(y)
  {
  }
};

struct Derived : Base
{
  using Base::Base;
};

int main()
{
  Derived d(3, 5);
  assert(d.a == 3);
  assert(d.b == 5);
  return 0;
}
