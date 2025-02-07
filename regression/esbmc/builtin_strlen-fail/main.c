#include <assert.h>
#ifndef __has_builtin        // Optional of course.
#  define __has_builtin(x) 0 // Compatibility with non-clang compilers.
#endif

int main()
{
#if __has_builtin(__builtin_strlen)
  const char src[9] = "testing!";
  assert(__builtin_strlen(src) == 88); // should be 8
  return 0;
#else
  __ESBMC_assert(0, "This test requires __builtin_strlen support");
#endif
}
