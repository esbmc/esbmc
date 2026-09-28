#include <string>
#include <cassert>

int main()
{
  std::string s("abcxx");
  // The last character not in "x" is 'c' at index 2, not 4.
  assert(s.find_last_not_of('x') == 4);
  return 0;
}
