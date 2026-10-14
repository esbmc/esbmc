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
  if (i < 1)
    assert(a.v == 2 && b.v == 2);
  else
    assert(a.v == 1 && b.v == 3);
}
