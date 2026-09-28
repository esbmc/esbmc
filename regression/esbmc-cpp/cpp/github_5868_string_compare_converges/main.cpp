#include <string>
#include <cassert>

int main()
{
  std::string a("abc");
  std::string b("abd");
  std::string c("abc");

  // [string.compare]: common prefix first, then the lengths.
  assert(a.compare(b) < 0);
  assert(b.compare(a) > 0);
  assert(a.compare(c) == 0);
  assert(a.compare("abc") == 0);
  assert(a.compare("abd") < 0);

  std::string longer("abcd");
  assert(a.compare(longer) < 0);
  assert(longer.compare(a) > 0);

  return 0;
}
