#include <assert.h>
#ifndef __has_builtin        // Optional of course.
#  define __has_builtin(x) 0 // Compatibility with non-clang compilers.
#endif

int main()
{
#if __has_builtin(__builtin_memcpy)
  const char src[9] = "testing!";
  char dest[9] = {'A'};
  __builtin_memcpy(dest + 1, src + 1, 9 - 1);
  assert(dest[0] == 'A');
  assert(dest[1] == 'e');
  assert(dest[2] == 's');
  assert(dest[3] == 't');
  assert(dest[4] == 'i');
  assert(dest[5] == 'n');
  assert(dest[6] == 'g');
  assert(dest[7] == '!');
  assert(dest[8] == '\0');
  return 0;
#endif
}
