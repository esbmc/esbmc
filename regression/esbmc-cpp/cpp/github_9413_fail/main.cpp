#include <cassert>
#include <string>

int main()
{
  std::string a("a\0b", 3), b("a\0c", 3);
  assert(a == b);
  return 0;
}
