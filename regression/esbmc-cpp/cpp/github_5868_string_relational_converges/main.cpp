#include <string>
#include <cassert>

int main()
{
  std::string a("abc");
  std::string b("abd");
  std::string c("abc");
  std::string longer("abcd");

  // [string.cmp]: all six are defined in terms of compare().
  assert(a < b);
  assert(!(b < a));
  assert(b > a);
  assert(a <= c);
  assert(a >= c);
  assert(a <= b);
  assert(b >= a);

  // A prefix orders before the longer string.
  assert(a < longer);
  assert(longer > a);
  return 0;
}
