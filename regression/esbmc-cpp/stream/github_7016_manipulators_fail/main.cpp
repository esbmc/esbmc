#include <sstream>
#include <iomanip>
#include <cassert>

// Pins github #7016: the pre-fix model discarded every insertion reached
// through an ostream&, so this empty-output claim held.
int main()
{
  std::ostringstream ss;
  ss << std::setw(4) << 7;
  assert(ss.str().size() == 0);
  return 0;
}
