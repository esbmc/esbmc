// github #6291: a reference bound to a conditional (ternary) lvalue must alias
// the selected operand. Writing through the reference updates that operand.
#include <cassert>

int main()
{
  int a = 1, b = 2;
  int &r = (a < b) ? a : b; // a<b true -> r binds to a
  r = 9;
  assert(a == 9 && b == 2);

  int c = 5, d = 3;
  int &s = (c < d) ? c : d; // c<d false -> s binds to d
  s = 7;
  assert(d == 7 && c == 5);
  return 0;
}
