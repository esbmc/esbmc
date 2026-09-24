// Specialisations differing only in a member-pointer argument must not share
// one symbol: clang's USR spells that argument as nothing.
#include <cassert>
template <class T> struct traits;
struct A { int b; };
struct B { long e; };
template <> struct traits<int A::*> { static const int v = 1; };
template <> struct traits<long B::*> { static const int v = 2; };
template <class T> struct W { int get() { return traits<T>::v; } };
int main() {
  W<int A::*> wa;
  W<long B::*> wb;
  assert(wa.get() == 1);
  assert(wb.get() == 2);
  return 0;
}
