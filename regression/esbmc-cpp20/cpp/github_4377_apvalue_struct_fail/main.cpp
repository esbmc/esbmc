// Failing companion to github_4377_apvalue_struct: the consteval-folded
// struct value still flows through APValue::Struct lowering, but the
// assertion reads back the wrong field value so verification must fail.
#include <cassert>

struct Pair
{
  int x;
  int y;
  consteval Pair(int a, int b) noexcept : x(a), y(b) {}
};

int main()
{
  Pair p{3, 4};
  assert(p.x == 99);
  return 0;
}
