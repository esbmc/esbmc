// [string.access] pairs charT& at(size_type) with const charT& at(size_type)
// const; only the const by-value form existed (#7567).
#include <cassert>
#include <string>

int main()
{
  std::string s = "abc";

  s.at(0) = 'z';
  assert(s[0] == 'z');

  char &r = s.at(1);
  r = 'y';
  assert(s[1] == 'q');

  const std::string &c = s;
  assert(c.at(2) == 'c');

  return 0;
}
