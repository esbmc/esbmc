// C++17 added the non-const charT* data() ([string.accessors]); the model had
// only the const overload (#7567).
#include <cassert>
#include <string>

int main()
{
  std::string s = "abc";

  char *p = s.data();
  p[0] = 'z';
  assert(s[0] == 'z');

  const char *q = s.data();
  assert(q[1] == 'b');

  return 0;
}
