#include <type_traits>

int main()
{
  // [meta.unary.prop.query]: the number of array dimensions.
  static_assert(std::rank<int>::value == 0, "scalar");
  static_assert(std::rank<int[]>::value == 1, "unbounded");
  static_assert(std::rank<int[3]>::value == 1, "bounded");
  static_assert(std::rank<int[3][4]>::value == 2, "two dimensions");
  static_assert(std::rank<int[3][4][5]>::value == 3, "three dimensions");
  static_assert(std::rank<int *>::value == 0, "pointer is not an array");
  static_assert(std::rank_v<int[2][2]> == 2, "_v alias");
  return 0;
}
