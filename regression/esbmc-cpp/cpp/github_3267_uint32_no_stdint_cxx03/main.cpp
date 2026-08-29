// Pre-C++11 counterpart of github_3267_uint32_no_stdint: <string> used to
// reach <cstdint> only through the C++11-guarded <string_view> (github #3267).
#include <sstream>
#include <string>
#include <cassert>

int main()
{
  const std::string s = "HI!";
  std::stringstream ss;
  uint32_t n = 0;
  ss << char('0' + n);
  assert(s.size() == 3);
  return 0;
}
