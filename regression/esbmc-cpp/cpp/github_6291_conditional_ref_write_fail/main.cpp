// Negative counterpart of github_6291_conditional_ref_write: with the condition
// nondeterministic, `b` is not always the arm that gets bumped, so pinning the
// write to one specific arm must fail.  Guards against a fix that makes the
// bound reference alias both arms at once.
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
  int c = nondet_int();

  bump((c < 1) ? a : b);
  assert(b.v == 12);

  return 0;
}
