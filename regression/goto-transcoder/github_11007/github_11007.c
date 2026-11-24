// #11007: _FORTIFY_SOURCE builtins as goto-cc emits them on macOS.
#include <assert.h>
#include <string.h>
int main()
{
  char a[4] = {1, 2, 3, 4}, b[4], c[4];
  __builtin___memcpy_chk(b, a, 4, __builtin_object_size(b, 0));
  assert(b[2] == 3);
  __builtin___memmove_chk(a + 1, a, 3, __builtin_object_size(a + 1, 0));
  assert(a[1] == 1 && a[3] == 3);
  __builtin___memset_chk(c, 7, 4, __builtin_object_size(c, 0));
  assert(c[3] == 7);
  char d[8], e[8];
  __builtin___strcpy_chk(d, "hi", __builtin_object_size(d, 0));
  assert(d[1] == 'i' && d[2] == 0);
  __builtin___strcat_chk(d, "yo", __builtin_object_size(d, 0));
  assert(d[3] == 'o' && d[4] == 0);
  __builtin___strncpy_chk(e, "ab", 3, __builtin_object_size(e, 0));
  assert(e[1] == 'b' && e[2] == 0);
  __builtin___strncat_chk(e, "cd", 1, __builtin_object_size(e, 0));
  assert(e[2] == 'c' && e[3] == 0);
  return 0;
}
