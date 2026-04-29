// github.com/esbmc/esbmc/issues/4190 — underlying_type gap.
// Verifies std::underlying_type<E>::type and underlying_type_t<E>
// resolve to the correct integer type for enum classes.

#include <type_traits>
#include <cassert>

enum class Color : unsigned char { red, green, blue };
enum class Priority : int { low, medium, high };

int main()
{
  static_assert(
    std::is_same_v<std::underlying_type_t<Color>, unsigned char>,
    "Color underlying type must be unsigned char");
  static_assert(
    std::is_same_v<std::underlying_type_t<Priority>, int>,
    "Priority underlying type must be int");
  static_assert(
    std::is_same_v<std::underlying_type<Color>::type, unsigned char>,
    "underlying_type<Color>::type must be unsigned char");
  return 0;
}
