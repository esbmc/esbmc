#include <string>
#include <cassert>

int main()
{
  std::string s("abcabc");
  // find returns the LOWEST matching position, so this is 1, not 4.
  assert(s.find('b') == 4);
  return 0;
}
