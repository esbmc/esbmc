#include <cassert>
#include <string>

int main()
{
  std::string s = "abc";

  // An empty needle matches at position 0, so this is not npos.
  assert(s.find("") == std::string::npos);

  return 0;
}
