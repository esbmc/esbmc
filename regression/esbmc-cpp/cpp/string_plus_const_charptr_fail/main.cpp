#include <string>
#include <cassert>

int main()
{
  std::string s("ab");
  const char *p = "cd";
  std::string r = s + p;
  assert(r.size() == 5);
  return 0;
}
