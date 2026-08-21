#include <string>
#include <cassert>

int main()
{
  std::string s("abcabc");

  // [string.find]: the lowest matching position at or after pos.
  assert(s.find('b') == 1);
  assert(s.find('b', 2) == 4);
  assert(s.find('z') == std::string::npos);
  assert(s.find("bc") == 1);
  return 0;
}
