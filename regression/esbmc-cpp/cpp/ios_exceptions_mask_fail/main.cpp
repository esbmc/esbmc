// Non-vacuity guard for ios_exceptions_mask: the mask is really stored, so
// asserting the wrong value must FAIL. Before the fix it was nondet and could
// not discriminate.
#include <sstream>
#include <cassert>

int main()
{
  std::ostringstream os;
  os.exceptions(std::ios_base::failbit);
  assert(os.exceptions() == std::ios_base::badbit);
  return 0;
}
