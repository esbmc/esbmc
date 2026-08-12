// The user's definition is what runs, so asserting the builtin's answer for it
// must be refutable -- otherwise the test above would pass for the wrong reason.
#include <cassert>

namespace mylib
{
int abs(int x)
{
  return x < 0 ? 0 - x : x + 1;
}
} // namespace mylib

int main()
{
  assert(mylib::abs(5) == 5);
  return 0;
}
