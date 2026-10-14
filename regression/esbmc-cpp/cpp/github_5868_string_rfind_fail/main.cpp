#include <string>
#include <cassert>

int main()
{
  std::string s("abc");
  // 'a' is at index 0 and nowhere else, so rfind is 0, not npos.
  assert(s.rfind('a') == std::string::npos);
  return 0;
}
