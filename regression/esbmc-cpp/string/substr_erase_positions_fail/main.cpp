#include <cassert>
#include <string>

int main()
{
  std::string s = "hello";

  // substr clamps the count to size() - pos, so this yields "ello", not the
  // five characters the assertion claims.
  assert(s.substr(1, 100).size() == 5);

  return 0;
}
