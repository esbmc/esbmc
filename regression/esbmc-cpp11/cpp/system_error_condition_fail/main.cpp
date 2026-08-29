// Non-vacuity guard for system_error_condition: a generic-category code does
// not match a system-category condition, since neither category claims the
// equivalence. Asserting it does must FAIL.
#include <system_error>
#include <cassert>

int main()
{
  std::error_code c(22, std::generic_category());
  std::error_condition d(22, std::system_category());
  assert(c == d);
  return 0;
}
