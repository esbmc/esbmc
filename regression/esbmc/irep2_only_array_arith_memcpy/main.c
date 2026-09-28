/* The reported shape: __builtin_memcpy over `dest + 1` / `src + 1` aborted
   under the sole-adjuster flag before the array operands were decayed. */
#include <assert.h>

int main(void)
{
  const char src[9] = "testing!";
  char dest[9] = {'A'};

  __builtin_memcpy(dest + 1, src + 1, 8);

  assert(dest[0] == 'A');
  assert(dest[8] == '\0');
  return 0;
}
