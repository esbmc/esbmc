// A write through a reference must reach what the reference names. IREP2
// spells a reference as a pointer carrying a reference kind, so an operand of
// that type has to be read through before it is used as a value -- otherwise
// `r = 7` casts 7 to the reference type and assigns `&7` to the reference
// itself, leaving the referent at its initial value
// (docs/roadmap/scope-clang-cpp-irep2.md §3.16).
#include <cassert>

int pick(int &r)
{
  r = 7;
  return r;
}

int main()
{
  int v = 0;
  int &r = v;

  r = 3;
  assert(v == 3);

  pick(v);
  assert(v == 7);

  const int sum = r + 1;
  assert(sum == 8);
}
