#include <string>
#include <cassert>

int main()
{
  std::string s("abcxx");

  // [string.find.last.not.of]
  assert(s.find_last_not_of('x') == 2);
  assert(s.find_last_not_of("x") == 2);

  std::string all("xxx");
  assert(all.find_last_not_of('x') == std::string::npos);
  return 0;
}
