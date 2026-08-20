#include <string>
#include <cassert>

int main()
{
  std::string a("abc");
  std::string longer("abcd");
  // [string.cmp]: a prefix orders before the longer string.
  assert(a > longer);
  return 0;
}
