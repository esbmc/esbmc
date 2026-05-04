// Baseline for esbmc#4270: with no -I override, <span> must still resolve
// the bundled <array> correctly.  Confirms the fix to use angle-bracket
// include does not regress the default lookup path.
#include <span>
#include <array>
#include <cassert>

int main()
{
  int buf[4] = {1, 2, 3, 4};
  std::span<int> s(buf, 4);
  std::array<int, 4> a{1, 2, 3, 4};
  assert(s[2] == a[2]);
  return 0;
}
