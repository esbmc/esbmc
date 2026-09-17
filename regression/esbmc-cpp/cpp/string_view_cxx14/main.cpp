#include <string_view>
#include <string>
#include <cassert>

/* The C++17 gate on char_traits::length must not cost C++14 the rest of
 * string_view: libc++ offers the header, the counted constructor as a constant
 * expression, and the string conversion in this mode. */
constexpr std::string_view COUNTED("abc", 3);

int main()
{
  static_assert(COUNTED.size() == 3, "constexpr counted ctor");

  std::string_view sv("@base@");
  assert(sv.size() == 6);

  std::string s(sv);
  assert(s.size() == 6);
  return 0;
}
