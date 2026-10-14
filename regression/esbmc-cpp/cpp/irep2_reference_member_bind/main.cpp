// A constructor's member initialiser *binds* a reference member; it must not be
// read through, or the constructor copies the referent and every later write
// through the member misses the original. The converter marks such an lhs with
// #member_init, which IREP2 carries on sideeffect_assign2t
// (docs/roadmap/scope-clang-cpp-irep2.md §3.16).
#include <cassert>

struct Holder
{
  int &r;
  Holder(int &x) : r(x)
  {
  }
};

int main()
{
  int v = 1;
  Holder h(v);
  h.r = 9;
  assert(v == 9);
}
