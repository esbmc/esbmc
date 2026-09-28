#include <cassert>
#include <string>

int main()
{
  std::string s("abc");

  // [string.capacity]: size() returns size_type, an unsigned integer type, so
  // the subtraction wraps instead of going negative.
  assert(s.size() - 5 > 0);
  assert(s.size() == s.length());

  std::string e;
  assert(e.size() - 1 > 0);

  return 0;
}
