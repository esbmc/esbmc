#include <cassert>
#include <string>

int main()
{
  std::string s("abcde");

  // [string.accessors]: c_str() is a null-terminated array of size() chars, so
  // shrinking has to move the terminator down to n.
  s.resize(2, 'x');
  assert(s.size() == 2);
  assert(s.c_str()[2] == '\0');
  assert(s == "ab");

  return 0;
}
