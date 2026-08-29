#include <string>
#include <cassert>

int main()
{
  std::string s("xxabc");
  // The first character not in "x" is at index 2, not 0.
  assert(s.find_first_not_of('x') == 0);
  return 0;
}
