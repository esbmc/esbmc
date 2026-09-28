// Negative counterpart of github_3267_uint32_no_stdint: the uint32_t typedef
// must be the real 32-bit unsigned type, not merely a name that parses.
#include <sstream>
#include <string>
#include <cassert>

int main()
{
  const std::string s = "HI!";
  std::stringstream ss;
  uint32_t n = 0;
  ss << char('0' + n);
  assert(sizeof(uint32_t) == 8);
  assert(s.size() == 3);
  return 0;
}
