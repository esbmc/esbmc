// Specialisations differing only in the type of a nullptr template argument
// must not share one symbol: clang's USR spells that argument as nothing.
#include <cassert>
template <class T> struct traits;
template <> struct traits<int *> { static const int v = 1; };
template <> struct traits<long *> { static const int v = 2; };
template <auto P> int g() { return traits<decltype(P)>::v; }
int main() {
  assert(g<(int *)nullptr>() == 2);
  assert(g<(long *)nullptr>() == 2);
  return 0;
}
