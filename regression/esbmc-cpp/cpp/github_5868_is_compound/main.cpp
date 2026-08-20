#include <type_traits>
struct S { int m; };
int main()
{
  static_assert(!std::is_compound<int>::value, "arithmetic");
  static_assert(!std::is_compound<void>::value, "void");
  static_assert(std::is_compound<S>::value, "class");
  static_assert(std::is_compound<int *>::value, "pointer");
  static_assert(std::is_compound<int &>::value, "reference");
  static_assert(std::is_compound<int[3]>::value, "array");
  static_assert(std::is_compound_v<S>, "_v alias");
  return 0;
}
