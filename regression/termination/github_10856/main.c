/* #10856: memcpy with a null source and length 0 runs __memcpy_impl's loop
 * zero times. The inductive step assumed the loop was entered, so no state
 * reached the end of main and the step read as non-termination. */
#include <string.h>

int main(void)
{
  char d[4];
  memcpy(d, 0, 0);
  unsigned x = 1;
  while (x != 0)
    x <<= 1;
  return 0;
}
