#include <cassert>
#include <string>

// [string.access] p1 requires the const overload; indexing a const reference
// is ubiquitous in codec/parser code.
static char at(const std::string &s, unsigned i)
{
  return s[i];
}

int main()
{
  std::string s("ab");
  s.reserve(16);
  s.push_back('c');

  assert(s.size() == 3);
  assert(at(s, 0) == 'a');
  assert(at(s, 1) == 'b');
  assert(at(s, 2) == 'c');

  return 0;
}
