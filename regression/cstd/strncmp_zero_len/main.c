#include <string.h>
#include <assert.h>

int main(void)
{
  /* C11/C17 7.24.4.4: strncmp compares "not more than n characters", so for
   * n == 0 zero characters are compared and the result is 0, regardless of
   * the buffer contents. */
  assert(strncmp("hello", "", 0) == 0);    /* differing length and content */
  assert(strncmp("abc", "abc", 0) == 0);   /* identical buffers */
  assert(strncmp("abc", "xyz", 0) == 0);   /* fully differing buffers */
  return 0;
}
