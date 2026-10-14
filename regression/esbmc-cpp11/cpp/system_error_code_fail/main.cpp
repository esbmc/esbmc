// Non-vacuity guard for system_error_code: codes from different categories
// really are distinct, so asserting they are equal must FAIL.
#include <system_error>
#include <cassert>

int main()
{
  std::error_code f(22, std::generic_category());
  std::error_code g(22, std::system_category());
  assert(f == g);
  return 0;
}
