// Writing through a reference parameter bound to a conditional lvalue must hit
// exactly the selected arm.  Before the fix the concrete case below bumped `b`
// instead of `a`.  The nondeterministic case additionally pins that the bound
// reference aliases one arm rather than both.
#include <cassert>

int nondet_int();

struct P
{
  int v;
};

void bump(P &x)
{
  x.v += 10;
}

int main()
{
  P a{1}, b{2};
  int c = 0;
  bump((c < 1) ? a : b);
  assert(a.v == 11);
  assert(b.v == 2);

  P p{1}, q{2};
  int n = nondet_int();
  bump((n < 1) ? p : q);
  assert((p.v == 11 && q.v == 2) || (p.v == 1 && q.v == 12));

  return 0;
}
