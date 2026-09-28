#include <string>
#include <cassert>

int main()
{
  std::string a("ab");
  std::string b("cd");
  std::string ab = a + b;
  // Concatenation keeps both operands, so the result is four characters.
  assert(ab.size() == 2);
  return 0;
}
