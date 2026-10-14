#include <cassert>
#include <string>

int main()
{
  // [string.capacity]: resize(n) is equivalent to resize(n, charT()), so the
  // appended positions are null characters, not whatever the buffer held.
  std::string s("abc");
  s.resize(5);
  assert(s.size() == 5);
  assert(s[3] == '\0');
  assert(s[4] == '\0');
  assert(s[0] == 'a' && s[1] == 'b' && s[2] == 'c');

  std::string t("ab");
  t.resize(4, 'x');
  assert(t.size() == 4);
  assert(t[2] == 'x');
  assert(t[3] == 'x');
  assert(t == "abxx");

  return 0;
}
