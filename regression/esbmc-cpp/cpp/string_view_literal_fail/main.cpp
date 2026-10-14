#include <string_view>
#include <string>
#include <cassert>

inline constexpr std::string_view PREFIX = "@base@";

int main()
{
  std::string s(PREFIX);
  assert(s.size() == 7);
  return 0;
}
