#include <cassert>
#include <string>

int main()
{
  const std::string a = "abc";
  const std::string b = "abd";
  std::string c = "abc";
  std::string d = "abd";

  assert(a < b);
  assert(b <= a);
  assert(b > a);
  assert(b >= a);
  assert(a <= a);
  assert(a >= a);

  assert(c < d);
  assert(c <= d);
  assert(d >= c);

  assert(a <= "abd");
  assert(a >= "abb");
  assert("abb" <= a);
  assert("abd" >= a);
  return 0;
}
