// A defaulted comparison declared as a friend takes both operands as
// parameters. Leaving them unnamed made the synthesised body compare unbound
// operands, so every == returned true.
#include <cassert>

struct P
{
  int v;
  friend bool operator==(P, P) = default;
};

struct Q
{
  int a;
  char b;
  bool c;
  friend bool operator==(Q, Q) = default;
};

int main()
{
  P a{1}, b{2};
  assert(a == a);
  assert(!(a == b));

  Q x{1, 'z', true}, y{1, 'z', true}, z{1, 'z', false};
  assert(x == y);
  assert(!(x == z));
  return 0;
}
