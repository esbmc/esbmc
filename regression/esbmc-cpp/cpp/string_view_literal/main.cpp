#include <string_view>
#include <string>
#include <cassert>

// [string.view]: basic_string_view is a literal type, so this is well-formed.
// base_subobject.h declares exactly this.
inline constexpr std::string_view PREFIX = "@base@";

int main()
{
  static_assert(PREFIX.size() == 6, "constexpr size");
  static_assert(PREFIX[0] == '@', "constexpr subscript");
  static_assert(!PREFIX.empty(), "constexpr empty");
  static_assert(PREFIX.data()[1] == 'b', "constexpr data");

  constexpr std::string_view counted("abc", 3);
  static_assert(counted.size() == 3, "constexpr counted ctor");

  // [string.cons]: the explicit string_view constructor.
  std::string s(PREFIX);
  assert(s.size() == 6);

  std::string_view cid("Foo");
  std::string joined = std::string(PREFIX) + std::string(cid);
  assert(joined.size() == 9);
  return 0;
}
