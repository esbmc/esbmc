#include <string>
#include <cassert>

int main()
{
  std::string s("abcabc");

  // [string.rfind]: the highest position at or before pos holding c.
  assert(s.rfind('a') == 3);
  assert(s.rfind('a', 2) == 0);
  assert(s.rfind('z') == std::string::npos);

  std::string one("abc");
  assert(one.rfind('a') == 0);

  // [string.find.last.of]
  assert(s.find_last_of('b') == 4);
  return 0;
}
