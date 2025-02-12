#include <assert.h>
#ifndef __has_builtin        // Optional of course.
#  define __has_builtin(x) 0 // Compatibility with non-clang compilers.
#endif

int main()
{
#if __has_builtin(__builtin_malloc)
  char *p = (char *)__builtin_malloc(5);
  for (int i = 0; i < 5; i++)
  {
    p[i] = 'a';
  }
  assert(p[0] == 'b'); // should be 'a'
  assert(p[1] == 'a');
  assert(p[2] == 'a');
  assert(p[3] == 'a');
  assert(p[4] == 'a');
  return 0;
#else
  __ESBMC_assert(0, "This test requires __builtin_malloc support");
#endif
}
