#include <cassert>
#include <string>

int main()
{
  std::string a = "AAAA", b = "AA";

  // Before the fix, operator> stopped at the end of the shorter string and
  // returned false for this strict-prefix comparison, so this (wrong) assertion
  // held. With the fix "AAAA" > "AA" is true, so the assertion now fails. This
  // pins the unsound direction of the bug.
  assert(!(a > b));

  return 0;
}
