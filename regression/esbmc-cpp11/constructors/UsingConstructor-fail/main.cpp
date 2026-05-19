#include <cassert>

struct Base
{
  int value;
  explicit Base(int v) : value(v)
  {
  }
};

struct Derived : Base
{
  using Base::Base;
};

int main()
{
  Derived d(42);
  assert(d.value == 7);
  return 0;
}
