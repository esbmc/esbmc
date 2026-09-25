/* pragma_unroll_count is excluded from the loop kinds' `fields` tuple, so it
   does not participate in equality: a rebuild that drops it compares equal to
   one that keeps it, and nothing in the pass notices. Dropped, the loop runs to
   its natural bound of 8 and writes a[3..7] out of a 3-element array -- a
   spurious bounds violation seen only under the flag. The while loop covers the
   condition-rebuild site, the for loop the init hoist. */
#include <stdint.h>

int main(void)
{
  int a[3] = {0};
  int b[3] = {0};

#pragma unroll 3
  for (uint32_t j = 0; j < 8; j++)
    a[j] = (int)(j + 1);

  uint32_t k = 0;
#pragma unroll 3
  while (k < 8)
  {
    b[k] = (int)(k + 1);
    k++;
  }

  return 0;
}
