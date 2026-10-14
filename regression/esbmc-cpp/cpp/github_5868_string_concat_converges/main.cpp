#include <string>
#include <cassert>

int main()
{
  std::string a("ab");
  std::string b("cd");

  std::string ab = a + b;
  assert(ab.size() == 4);
  assert(ab[0] == 'a');
  assert(ab[3] == 'd');

  std::string ac = a + 'z';
  assert(ac.size() == 3);
  assert(ac[2] == 'z');
  return 0;
}
