#include <cassert>

// `s.*pmf` and `p->*pmf` carry clang's BoundMember type, which has no IREP2
// form: the selection has to become the member function itself, with `this`
// prepended to its parameters, or goto_convert sees a callee it cannot read.
struct S
{
  int v;
  int get() const
  {
    return v;
  }
};

int main()
{
  S s;
  s.v = 7;
  int (S::*pmf)() const = &S::get;
  assert((s.*pmf)() == 7);

  S *p = &s;
  assert((p->*pmf)() == 7);
}
