// Negative companion to github_6293_member_fn_ptr: the call binds `this` and
// the argument correctly, so g uses v (100) + x (5) == 105; asserting a wrong
// value is violated.
#include <cassert>

struct S
{
  int v;
  int g(int x) { return v + x; }
};

int main()
{
  S s;
  s.v = 100;
  int (S::*p)(int) = &S::g;
  assert((s.*p)(5) == 999); // wrong on purpose
  return 0;
}
