#include <string>
#include <cassert>

int main()
{
  std::string a("abc");
  std::string b("xy");

  // Copying one std::string into another used to scan to a NUL over a
  // symbolic buffer and never terminate.
  a = b;
  assert(a.size() == 2);
  assert(a[0] == 'x');
  assert(a[1] == 'y');
  assert(a.c_str()[2] == '\0');

  // Self-assignment.
  b = b;
  assert(b.size() == 2);

  // Assigning a longer string over a shorter one.
  std::string c("z");
  c = a;
  assert(c.size() == 2);
  assert(c[1] == 'y');
  return 0;
}
