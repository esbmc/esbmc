#include <cassert>
#include <string>

static std::string tail(const std::string &s)
{
  return s.substr(1, 3);
}

int main()
{
  const std::string s = "hello";
  assert(tail(s) == "xxx");
  return 0;
}
