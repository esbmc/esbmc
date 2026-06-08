// github.com/esbmc/esbmc/issues/4190 — type_traits gap.
// Verifies is_enum / is_signed / is_unsigned / is_trivially_copyable
// (and their _v aliases) compile and behave correctly.

#include <type_traits>
#include <cassert>

enum class Color { red, green, blue };
struct PlainStruct { int a; double b; };
struct NonTrivial { NonTrivial() {} ~NonTrivial() {} int x; };

template <typename T>
typename std::enable_if<std::is_enum_v<T>, int>::type kind() { return 1; }

template <typename T>
typename std::enable_if<!std::is_enum_v<T>, int>::type kind() { return 0; }

int main()
{
  static_assert(std::is_enum_v<Color>);
  static_assert(!std::is_enum_v<int>);

  static_assert(std::is_unsigned_v<unsigned>);
  static_assert(!std::is_unsigned_v<int>);

  static_assert(std::is_signed_v<int>);
  static_assert(!std::is_signed_v<unsigned>);

  static_assert(std::is_trivially_copyable_v<PlainStruct>);
  static_assert(!std::is_trivially_copyable_v<NonTrivial>);

  assert(kind<Color>() == 1);
  assert(kind<int>() == 0);
  return 0;
}
