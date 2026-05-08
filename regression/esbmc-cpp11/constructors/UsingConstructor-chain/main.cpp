#include <cassert>

struct A
{
  int v;
  explicit A(int x) : v(x)
  {
  }
};

struct B : A
{
  using A::A;
};

struct C : B
{
  using B::B;
};

int main()
{
  C c(7);
  assert(c.v == 7);
  return 0;
}
