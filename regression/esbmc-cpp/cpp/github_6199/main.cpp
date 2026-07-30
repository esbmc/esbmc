#include <cassert>
#include <cstring>
#include <string>

// [string.cons] constructs a basic_string from the range [s, s + n). n is
// unrelated to strlen(s), and the range may contain embedded null characters.
int main()
{
  const char *lit = "AAAA=";

  // The exact-length form string(s, strlen(s)) is legal and must be accepted.
  std::string a(lit, 5);
  assert(a.length() == 5);
  assert(a[0] == 'A');
  assert(a[4] == '=');

  std::string b(lit, std::strlen(lit));
  assert(b.length() == 5);

  // Embedded null characters inside the range are preserved, and length()
  // agrees with the copied contents rather than stopping at the first '\0'.
  const char buf[6] = {'a', 'b', '\0', 'c', 'd', '\0'};
  std::string c(buf, 5);
  assert(c.length() == 5);
  assert(c[0] == 'a');
  assert(c[3] == 'c');
  assert(c[4] == 'd');

  // n == 0 is permitted and yields an empty string.
  std::string d("ignored", 0);
  assert(d.length() == 0);

  return 0;
}
