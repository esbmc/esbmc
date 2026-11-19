/* #10856: __memset_impl's loop writes through a pointer, so its havoc must be
 * skipped, not only its entry assumption, when the loop is not entered. */
#include <string.h>

int main(void)
{
  char *p = 0;
  memset(p, 0, 0);
  unsigned x = 1;
  while (x != 0)
    x <<= 1;
  return 0;
}
