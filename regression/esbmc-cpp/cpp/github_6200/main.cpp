#include <cassert>
#include <string>

int main()
{
  std::string s(10, 'x');
  assert(s.length() == 10);
  assert(s[0] == 'x');
  assert(s[9] == 'x');

  // [string.cons] permits n == 0
  std::string empty(0, 'y');
  assert(empty.length() == 0);

  // Largest fill the model admits: str[127] holds the terminator, so the
  // capacity assert must not reject this. Pins the bound against off-by-one.
  std::string full(127, 'z');
  assert(full.length() == 127);
  assert(full[126] == 'z');

  return 0;
}
