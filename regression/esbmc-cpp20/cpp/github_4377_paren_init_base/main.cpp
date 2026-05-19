#include <cassert>

struct B
{
  int b;
};

struct D : B
{
  int d;
};

int main()
{
  // Parenthesised aggregate init with a base-class initialiser; the base
  // sub-object's fields must flatten into D's components in declaration
  // order.
  D x(B{1}, 2);
  assert(x.b == 1);
  assert(x.d == 2);
  return 0;
}
