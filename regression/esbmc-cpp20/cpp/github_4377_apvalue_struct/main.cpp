// Exercise the APValue::Struct lowering added in clang_c_convertert::
// get_APValue_expr (PR #4386).  A `consteval` constructor wraps the
// CXXConstructExpr in a ConstantExpr whose APValueResult is
// APValue::Struct with N fields, which is the same shape that
// std::strong_ordering category values produce.
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
  assert(p.x == 3);
  assert(p.y == 4);
  return 0;
}
