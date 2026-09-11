// Incrementing through a reference returned by a function must update the
// referent. The increment updates its operand in place, so a reference-typed
// operand has to be read through first -- otherwise the arithmetic lands on the
// reference itself and the referent never changes
// (docs/roadmap/scope-clang-cpp-irep2.md §3.16).
//
// It has to be a reference-*returning call*: a local `int &r` is dereferenced
// at conversion time, so `r++` is already correct and pins nothing.
#include <cassert>

struct Box
{
  int v;
  int &at()
  {
    return v;
  }
};

int main()
{
  Box b;
  b.v = 0;

  b.at()++;
  assert(b.v == 0);

  ++b.at();
  assert(b.v == 2);

  b.at()--;
  assert(b.v == 1);
}
