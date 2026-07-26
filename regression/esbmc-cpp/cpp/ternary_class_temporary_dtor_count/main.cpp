// KNOWNBUG: a conditional operator with class-typed operands over-destroys.
//
// In C++17 `cond ? C(1) : C(2)` is a prvalue and `a` is initialised directly
// from it ([dcl.init]/17.6.1, guaranteed elision), so exactly one C object
// exists and exactly one destructor runs. ESBMC materialises a temporary per
// branch plus a result temporary, assigns between them bitwise, and then
// destroys all of them -- including the branch temporary that was never
// constructed on the taken path. dtors ends up > 1.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <cassert>

int dtors = 0;

struct C
{
  int v;
  explicit C(int x) : v(x)
  {
  }
  C(const C &o) : v(o.v)
  {
  }
  ~C()
  {
    ++dtors;
  }
};

int main()
{
  int n = 2;
  {
    C a = (n > 0) ? C(1) : C(2);
    assert(a.v == 1);
  }
  assert(dtors == 1);
  return 0;
}
