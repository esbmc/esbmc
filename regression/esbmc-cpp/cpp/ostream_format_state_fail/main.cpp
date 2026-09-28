// Non-vacuity guard for ostream_format_state: the setting is really stored, so
// asserting the wrong value must FAIL. Before the fix the setter was a no-op
// and width() was nondet, so this could not discriminate.
#include <sstream>
#include <cassert>

int main()
{
  std::ostringstream os;
  os.precision(3);
  assert(os.precision() == 4);
  return 0;
}
