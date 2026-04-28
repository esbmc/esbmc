// Negative test: assert a wrong underlying type so ESBMC produces VERIFICATION FAILED.

#include <cassert>
#include <type_traits>

enum class Color : unsigned char { red, green, blue };

int main()
{
  // underlying_type_t<Color> is unsigned char, NOT int — assertion must fail
  assert((std::is_same_v<std::underlying_type_t<Color>, int>));
  return 0;
}
