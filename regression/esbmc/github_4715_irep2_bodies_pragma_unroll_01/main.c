/* Pins that the --irep2-bodies body round-trip preserves a loop's
 * `#pragma unroll N` count. The pragma truncates this loop to 3 iterations,
 * so only a[0..2] are written -- all in bounds. If the count were dropped on
 * the round-trip the loop would run to its natural bound of 8, writing a[3..7]
 * out of the 3-element array: a spurious array-bounds violation seen only
 * under the flag. (An unsound unroll assumes loop exit after N iterations, so
 * a post-loop assertion would be vacuous; the in-loop bounds checks are the
 * discriminator.) */
#include <stdint.h>

int main()
{
  int a[3] = {0};

  #pragma unroll 3
  for (uint32_t j = 0; j < 8; j++)
    a[j] = (int)(j + 1);

  return 0;
}
