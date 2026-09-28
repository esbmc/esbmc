// A pseudo-destructor call does nothing but evaluate its base
// ([expr.pseudo]/1). The base's side effects must survive, and the node itself
// must not reach the goto program: IREP2 has no kind for it, so migrating one
// aborts (docs/roadmap/scope-clang-cpp-irep2.md §3.15).
#include <cassert>

typedef int scalar;

static int calls = 0;

scalar &base(scalar &x)
{
  calls++;
  return x;
}

int main()
{
  scalar v = 22;
  base(v).~scalar();
  assert(v == 22 && calls == 0);
}
