#include <cassert>
#include <string>

static char at(const std::string &s, unsigned i)
{
  return s[i];
}

int main()
{
  std::string s("ab");
  s.push_back('c');
  // Must fail: push_back appended 'c', so the const operator[] reads 'c'.
  assert(at(s, 2) == 'z');
  return 0;
}
