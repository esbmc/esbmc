#include <type_traits>
#include <map>
#include <cassert>

struct S
{
  int m;
};

int main()
{
  // [meta.unary.comp]
  static_assert(std::is_scalar<int>::value, "arithmetic");
  static_assert(std::is_scalar<int *>::value, "pointer");
  static_assert(std::is_scalar<int S::*>::value, "member pointer");
  static_assert(std::is_scalar<const double>::value, "cv-qualified");
  static_assert(!std::is_scalar<S>::value, "class");
  static_assert(!std::is_scalar<int &>::value, "reference");
  static_assert(std::is_scalar_v<int>, "_v alias");

  // [associative.reqmts]: the ordered lookups have const overloads.
  std::map<int, int> m;
  m[2] = 20;
  const std::map<int, int> &cm = m;
  assert(cm.lower_bound(2) != cm.end());
  assert(cm.upper_bound(2) == cm.end());
  return 0;
}
