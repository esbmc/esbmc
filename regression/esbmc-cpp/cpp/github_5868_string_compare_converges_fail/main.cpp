#include <string>
#include <cassert>

int main()
{
  std::string a("abc");
  std::string b("abd");
  // 'c' < 'd' at the first differing position, so this is negative.
  assert(a.compare(b) > 0);
  return 0;
}
