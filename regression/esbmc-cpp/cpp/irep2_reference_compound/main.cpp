// A compound assignment through a reference returned by a function. The
// reference must be read through before the update, or the arithmetic lands on
// the reference (docs/roadmap/scope-clang-cpp-irep2.md §3.16).
//
// A local `int &r` would not pin this: it is dereferenced at conversion time.
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
  b.v = 1;

  b.at() += 2;
  assert(b.v == 3);

  b.at() *= 4;
  assert(b.v == 12);
}
