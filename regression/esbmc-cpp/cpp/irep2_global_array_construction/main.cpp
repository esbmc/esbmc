#include <cassert>

int marker = 1;

// A class-typed array with static storage is constructed by
// clang_cpp_maint::adjust_init, which runs after the adjust pass and keys on the
// converter's `#constructor` marker. A body written back without it loses the
// construction: element 0 lands on a temporary and the rest stay
// nondeterministic.
struct R
{
  int *p;
  R() : p(&marker)
  {
  }
};

R g[2];

int main()
{
  static R s[2];
  assert(*g[0].p == 1 && *g[1].p == 1);
  assert(*s[0].p == 1 && *s[1].p == 1);
}
