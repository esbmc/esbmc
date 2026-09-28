#include <string>
#include <cassert>

int main()
{
  const char *x = "abc";
  std::string s(x ? x : "");
  // The source has three characters, not zero.
  assert(s.empty());
  return 0;
}
