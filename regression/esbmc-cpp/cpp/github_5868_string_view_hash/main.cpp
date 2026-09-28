// github #5868 gaps 2/3: std::string had no conversion to std::string_view, and
// std::hash<std::string_view> bound the deleted primary template. [string.view.hash]
// requires a view's hash to equal the corresponding string's, so both must use
// the same scheme.
#include <string>
#include <string_view>
#include <functional>
#include <cassert>

int main()
{
  std::string s = "hello";

  // [string.accessors]: operator basic_string_view
  std::string_view v = s;
  assert(v.size() == 5);
  assert(v[0] == 'h');
  assert(v.data() == s.c_str());

  assert(std::hash<std::string_view>{}(v) == std::hash<std::string>{}(s));

  std::string_view w("hello");
  assert(std::hash<std::string_view>{}(w) == std::hash<std::string>{}(s));

  std::string other = "world";
  assert(
    std::hash<std::string_view>{}(std::string_view(other)) !=
    std::hash<std::string_view>{}(v));

  return 0;
}
