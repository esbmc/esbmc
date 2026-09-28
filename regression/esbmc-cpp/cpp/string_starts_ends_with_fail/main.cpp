#include <string>
#include <cassert>

int main()
{
  std::string s = "hello";
  assert(s.starts_with("lo")); // "lo" is a suffix, not a prefix
  return 0;
}
