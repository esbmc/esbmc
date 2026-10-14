#include <string>
#include <cassert>

int main()
{
  std::string a("abc");
  std::string b("xy");
  a = b;
  // Assignment replaces the contents, so a is "xy" and its size is 2.
  assert(a.size() == 3);
  return 0;
}
