#include <string.h>
#include <assert.h>

int main(void)
{
  char a[3] = {'a', 'b', 'c'};
  char b[3] = {'x', 'y', 'z'};

  /* Zero-length compare starting one-past-the-end of each buffer. A correct
   * strncmp examines no characters, so it must not dereference a[3]/b[3].
   * The old model read s1[0]/s2[0] before testing the count, which is an
   * out-of-bounds read here and trips the dereference (bounds) check. */
  assert(strncmp(a + 3, b + 3, 0) == 0);
  return 0;
}
