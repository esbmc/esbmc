// Non-vacuity guard for any_copy: the copy really carries the held value, so a
// wrong expectation must FAIL.
#include <any>
#include <cassert>

int main()
{
  std::any a = 5;
  std::any b = a;
  assert(std::any_cast<int>(b) == 6);
  return 0;
}
