#include <cassert>

// A variadic constructor supplies more arguments than its parameter list
// declares, so a fold that keys on the arity declines it and the first argument
// is then converted against the `this` parameter.
struct T
{
  int x;
  T(int a, ...) : x(a)
  {
  }
};

struct L
{
  T m;
  L() : m(4, 5, 6)
  {
  }
};

int main()
{
  L l;
  assert(l.m.x == 4);
}
