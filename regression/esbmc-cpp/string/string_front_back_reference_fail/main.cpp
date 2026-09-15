// [string.access] pairs charT& front() with const charT& front() const, and
// likewise for back(); only the const by-value forms existed (#7537).
#include <cassert>
#include <string>

int main()
{
  std::string s = "abc";

  s.front() = 'y';
  s.back() = 'z';
  assert(s[0] == 'y');
  assert(s[2] == 'q');

  char &r = s.back();
  r = 'w';
  assert(s[2] == 'w');

  const std::string &c = s;
  assert(c.front() == 'y');

  return 0;
}
