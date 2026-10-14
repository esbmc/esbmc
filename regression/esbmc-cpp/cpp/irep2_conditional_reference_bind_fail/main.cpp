#include <cassert>

extern "C" int nondet_int();

// A conditional over same-typed lvalues is itself an lvalue ([expr.cond]/4), so
// the address taken for the reference parameter has to be taken per arm. Left as
// `&(c ? a : b)` the pointer analysis resolves neither arm and the write lands
// nowhere.
struct S
{
  int v;
  S(int p) : v(p)
  {
  }
};

void bump(S &r)
{
  r.v++;
}

int main()
{
  S a(1), b(2);
  int i = nondet_int();
  bump(i < 1 ? a : b);
  // The negative half asserts that *neither* object moved, which is false
  // whichever arm is selected -- and true if the write lands nowhere.
  assert(a.v == 1 && b.v == 2);
}
