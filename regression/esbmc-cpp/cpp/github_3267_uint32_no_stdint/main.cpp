// <string> and <sstream> must make the <cstdint> fixed-width typedefs visible,
// as libstdc++/libc++ do transitively; requiring an explicit <stdint.h> broke
// unmodified third-party sources (github #3267).
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
