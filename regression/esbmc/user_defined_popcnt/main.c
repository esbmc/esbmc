/* A program's own definition of a name ESBMC lowers to a builtin wins over
 * the lowering (#6904). __popcnt is MSVC's intrinsic spelling; clang accepts a
 * definition of it on other targets. */
#include <assert.h>

unsigned __popcnt(unsigned x)
{
  return x + 41;
}

int main()
{
  assert(__popcnt(1) == 42);
  return 0;
}
