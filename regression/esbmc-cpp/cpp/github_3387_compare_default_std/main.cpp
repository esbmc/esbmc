// <compare> is a C++20 header, but libstdc++ and libc++ both let a translation
// unit include it unconditionally in an older mode, where it expands to
// nothing.  ESBMC's model used to be unguarded, so this program failed to parse
// under the default language mode (github #3387).
#include <compare>
#include <cassert>

int main()
{
  int a = 1, b = 2;
  assert(a < b);
  return 0;
}
