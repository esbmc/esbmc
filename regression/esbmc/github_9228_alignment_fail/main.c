/* github #9228: alignment constrains only as many low bits as it names, so a
 * 32-byte-aligned object need not be 64-byte aligned. */
#include <assert.h>
#include <stdint.h>

_Alignas(32) char buf[3];

int main(void)
{
  assert((uintptr_t)buf % 64 == 0);
  return 0;
}
