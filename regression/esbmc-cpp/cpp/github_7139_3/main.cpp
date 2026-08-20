#include <cassert>
#include <string>

int main()
{
  std::string s("abcde");

  // [string.capacity]: n <= size() erases the last size() - n elements, and
  // [string.accessors] makes c_str() a null-terminated array of size() chars.
  s.resize(2);
  assert(s.size() == 2);
  assert(s[0] == 'a');
  assert(s[1] == 'b');
  assert(s.c_str()[2] == '\0');

  return 0;
}
