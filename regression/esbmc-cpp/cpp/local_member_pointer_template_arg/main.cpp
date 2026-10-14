// A class template specialised on member pointers into same-named local
// classes of two functions gives two distinct specialisations.
#include <cassert>
#include <cstring>
template <class T> struct Box;
template <class C, class M> struct Box<M C::*> { C obj; };
int f() { struct S { int a; }; Box<int S::*> b; b.obj.a = 1; return b.obj.a; }
int g()
{
  struct S { int a; int b; int c; };
  Box<int S::*> b;
  std::memset(&b, 0, sizeof b);
  ((int *)&b.obj)[2] = 7;
  return b.obj.c;
}
int main() { assert(f() == 1); assert(g() == 7); return 0; }
