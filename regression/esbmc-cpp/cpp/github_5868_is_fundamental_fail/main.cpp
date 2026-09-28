// esbmc/esbmc#5868 negative control: <type_traits> had no is_fundamental, which fmt's format.h
// reaches for. Fundamental types are the arithmetic types, cv void and cv
// std::nullptr_t ([basic.types.general], [meta.unary.comp] Table 53).
#include <cassert>
#include <cstddef>
#include <type_traits>

struct s
{
  int x;
};
enum e
{
  a
};

int main()
{
  static_assert(std::is_fundamental<int>::value, "");
  static_assert(std::is_fundamental<const double>::value, "");
  static_assert(std::is_fundamental<volatile char>::value, "");
  static_assert(std::is_fundamental<void>::value, "");
  static_assert(std::is_fundamental<const void>::value, "");
  static_assert(std::is_fundamental<std::nullptr_t>::value, "");
  static_assert(std::is_fundamental_v<bool>, "");

  static_assert(!std::is_fundamental<int *>::value, "");
  static_assert(!std::is_fundamental<int &>::value, "");
  static_assert(!std::is_fundamental<int[4]>::value, "");
  static_assert(!std::is_fundamental<s>::value, "");
  static_assert(!std::is_fundamental<e>::value, "");

  static_assert(std::is_null_pointer_v<std::nullptr_t>, "");
  static_assert(!std::is_null_pointer_v<void *>, "");

  assert(std::is_fundamental<int>::value);
  assert(std::is_fundamental<s>::value);
  return 0;
}
