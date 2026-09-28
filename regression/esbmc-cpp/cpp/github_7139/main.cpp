#include <cassert>
#include <string>

int main()
{
  std::string s("abc");

  // [string.capacity]: n == 0 is in range; it erases every element.
  s.resize(0);
  assert(s.size() == 0);
  assert(s.empty());
  assert(s.c_str()[0] == '\0');

  return 0;
}
