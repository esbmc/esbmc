// Regression for esbmc#4270: bundled <span> must use angle-bracket include
// for <array> so a user -I stub directory can override std::array.  Before
// the fix, <span> did `#include "array"` which always pulled in ESBMC's
// bundled array, conflicting with the stub and producing a redefinition.
#include <span>
#include <array>

int main()
{
  int buf[3] = {7, 8, 9};
  std::span<int> s(buf, 3);
  std::array<int, 3> a;
  a[0] = s[0];
  return 0;
}
