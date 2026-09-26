// A class template specialised on same-named local classes of two functions
// gives two distinct specialisations.
#include <cassert>
template <class T> struct Box { T t; };
int f() { struct S { long a; }; Box<S> b; b.t.a = 300; return (int)b.t.a; }
int g() { struct S { char a; }; Box<S> b; b.t.a = 44; return b.t.a; }
int main() {
  assert(g() == 44);
  assert(f() == 44);
  return 0;
}
