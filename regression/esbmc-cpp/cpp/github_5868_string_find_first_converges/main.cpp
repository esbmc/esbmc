#include <string>
#include <cassert>

int main()
{
  std::string s("xxabc");

  // [string.find.first.not.of] / [string.find.first.of]
  assert(s.find_first_not_of('x') == 2);
  assert(s.find_first_of('a') == 2);
  assert(s.find_first_not_of("x") == 2);
  assert(s.find_first_of("ba") == 2);
  return 0;
}
