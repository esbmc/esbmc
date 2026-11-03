/* github #9228: an object's alignment is a power of two, so its base address
 * is constrained by zeroing its low bits rather than by a bvurem. */
#include <stdint.h>
#include <stdlib.h>

_Alignas(32) char buf[3];

int main(void)
{
  int *p = malloc(sizeof(int));
  __ESBMC_assert((uintptr_t)buf != (uintptr_t)p, "distinct objects");
  return 0;
}
