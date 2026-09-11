// A write through a reference returned by a function must reach the referent.
// IREP2 spells a reference as a pointer carrying a reference kind, so such an
// operand has to be read through before it is used as a value -- otherwise
// `at() = 7` casts 7 to the reference type and assigns `&7` to the returned
// pointer, leaving the referent at its initial value
// (docs/roadmap/scope-clang-cpp-irep2.md §3.16).
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
  b.at() = 7;
  assert(b.v == 7);
}
