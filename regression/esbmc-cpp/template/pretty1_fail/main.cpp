#include <cassert>
// Negative variant of pretty1: asserts the GCC-style __PRETTY_FUNCTION__
// format. ESBMC uses the Clang frontend, which emits the "[T = void]"
// form instead of GCC's "[with T = void]"; this assertion must fail.

#include <string.h>

template <typename T>
struct X
{
  X() { assert(strcmp(__PRETTY_FUNCTION__, "X<T>::X() [with T = void]") == 0); }
};

int main()
{
  X<void> x;
  return 0;
}
